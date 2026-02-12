from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_project_on_sys_path() -> None:
    root = _project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="query")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--collection", default=None)
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed RAG pipeline steps"
    )
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--config", default="config/settings.yaml")
    return parser.parse_args(argv)


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _format_snippet(text: str, *, max_len: int = 120) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= max_len:
        return compact
    return compact[: max(0, max_len - 1)] + "…"


def _extract_page(metadata: Dict[str, Any]) -> Optional[str]:
    for key in ("page", "page_number", "page_no", "pageno"):
        v = metadata.get(key)
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return None


def _print_stage(emoji: str, title: str) -> None:
    print(f"\n{emoji}  {title}")


def _print_ranked_items(items: List[Dict[str, Any]], *, top_k: int) -> None:
    for i, it in enumerate(items[:top_k], start=1):
        score = it.get("score")
        score_str = f"{float(score):.4f}" if isinstance(score, (int, float)) else "-"
        metadata = it.get("metadata")
        metadata_dict = metadata if isinstance(metadata, dict) else {}
        source = Path(
            str(metadata_dict.get("source_path") or metadata_dict.get("source") or "-")
        ).name
        page = _extract_page(metadata_dict) or "-"
        text = str(it.get("text", "") or "")
        print(
            f"  [{i:02d}] 🏆 {score_str} | 📄 {source} (p.{page}) | ID: {it.get('chunk_id','-')[:8]}...\n"
            f"       {_format_snippet(text)}"
        )


def _print_exception_chain(err: BaseException) -> None:
    msgs: list[str] = []
    cur: BaseException | None = err
    while cur is not None:
        msg = str(cur).strip() or cur.__class__.__name__
        if msg not in msgs:
            msgs.append(msg)
        cur = cur.__cause__
    for i, m in enumerate(msgs):
        prefix = "ERROR" if i == 0 else "CAUSE"
        print(f"{prefix}: {m}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    _ensure_project_on_sys_path()

    args = _parse_args(list(argv) if argv is not None else sys.argv[1:])
    query = str(args.query or "").strip()
    if not query:
        print("ERROR: --query 不能为空", file=sys.stderr)
        return 2

    from src.core.query_engine.dense_retriever import DenseRetriever
    from src.core.query_engine.fusion import RRFFusion
    from src.core.query_engine.hybrid_search import (
        _resolve_record_from_dense_vector_store,
    )
    from src.core.query_engine.query_processor import QueryProcessor
    from src.core.query_engine.reranker import Reranker
    from src.core.query_engine.sparse_retriever import SparseRetriever
    from src.core.settings import load_settings
    from src.core.trace.trace_context import TraceContext
    from src.observability.logger import write_trace

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = _project_root() / config_path

    try:
        settings = load_settings(str(config_path))
        if args.collection:
            settings.vector_store.collection_name = str(args.collection)
    except Exception as e:
        _print_exception_chain(e)
        return 1

    # 1. Check Environment
    if str(settings.vector_store.backend).lower() == "chroma" and not _module_available(
        "chromadb"
    ):
        print("ERROR: chromadb not installed.", file=sys.stderr)
        return 1

    # 2. Process Query
    if args.verbose:
        _print_stage("🔍", f"Processing Query: '{query}'")

    trace = TraceContext(trace_type="query")

    qp = QueryProcessor()
    effective_query = query
    if args.collection and "collection:" not in effective_query:
        effective_query = f"collection:{args.collection} {effective_query}".strip()

    qp_start = time.time() * 1000.0
    processed = qp.process(effective_query)
    qp_end = time.time() * 1000.0
    
    trace.record_stage(
        "query_processing",
        start_ms=qp_start,
        end_ms=qp_end,
        data={
            "original_query": query,
            "effective_query": effective_query,
            "keywords": processed.keywords,
            "filters": processed.filters,
        },
    )

    filters = dict(processed.filters or {})
    if args.collection:
        filters.setdefault("collection", str(args.collection))

    if args.verbose:
        print(f"   - Keywords: {processed.keywords}")
        print(f"   - Filters: {filters}")

    try:
        dense = DenseRetriever(settings)
        sparse = SparseRetriever(settings)
        fusion = RRFFusion()

        dense_top_k = int(settings.retrieval.top_k_dense)
        sparse_top_k = int(settings.retrieval.top_k_sparse)
        final_top_k = (
            int(args.top_k)
            if args.top_k is not None
            else int(settings.retrieval.top_k_final)
        )

        sparse_query = " ".join(processed.keywords).strip() or effective_query

        # 3. Dense Retrieval
        if args.verbose:
            _print_stage("🧠", "Dense Retrieval (Vector Search)")
        
        dense_start = time.time() * 1000.0
        dense_hits = dense.retrieve(effective_query, filters=filters, top_k=dense_top_k, trace=trace)
        dense_end = time.time() * 1000.0
        trace.record_stage(
            "dense",
            start_ms=dense_start,
            end_ms=dense_end,
            metrics={"n_hits": float(len(dense_hits))},
            data={"query": effective_query, "top_k": dense_top_k},
        )

        if args.verbose:
            max_score = f"{dense_hits[0].score:.4f}" if dense_hits else "N/A"
            print(f"   - Found {len(dense_hits)} candidates. Max score: {max_score}")

        # 4. Sparse Retrieval
        if args.verbose:
            _print_stage("🔡", "Sparse Retrieval (Keyword Search)")
        
        sparse_start = time.time() * 1000.0
        sparse_hits = sparse.retrieve(
            sparse_query,
            filters=filters,
            top_k=sparse_top_k,
            collection=str(args.collection) if args.collection else None,
            trace=trace,
        )
        sparse_end = time.time() * 1000.0
        trace.record_stage(
            "sparse",
            start_ms=sparse_start,
            end_ms=sparse_end,
            metrics={"n_hits": float(len(sparse_hits))},
            data={"query": sparse_query, "top_k": sparse_top_k},
        )

        if args.verbose:
            print(f"   - Found {len(sparse_hits)} candidates.")

        # 5. Fusion
        if args.verbose:
            _print_stage("🔀", "RRF Fusion (Hybrid Search)")
        need_candidates = final_top_k
        if not args.no_rerank:
            need_candidates = max(
                need_candidates, int(getattr(settings.rerank, "top_m", need_candidates))
            )

        fusion_start = time.time() * 1000.0
        fused_hits = fusion.fuse(dense_hits, sparse_hits, top_k=need_candidates)
        fusion_end = time.time() * 1000.0
        trace.record_stage(
            "fusion",
            start_ms=fusion_start,
            end_ms=fusion_end,
            metrics={
                "n_output": float(len(fused_hits)),
                "n_input": float(len(dense_hits) + len(sparse_hits)),
            },
        )

        if args.verbose:
            print(f"   - Combined into {len(fused_hits)} candidates.")

        # Hydrate
        dense_by_id = {
            h.record.id: h.record for h in dense_hits if getattr(h, "record", None)
        }
        hydrated: List[Dict[str, Any]] = []
        for fh in fused_hits:
            chunk_id = getattr(fh, "chunk_id", None)
            if not isinstance(chunk_id, str):
                continue
            record = dense_by_id.get(
                chunk_id
            ) or _resolve_record_from_dense_vector_store(dense, chunk_id)
            if record:
                hydrated.append(
                    {
                        "chunk_id": chunk_id,
                        "text": record.content,
                        "metadata": dict(record.metadata or {}),
                        "score": float(getattr(fh, "score", 0.0)),
                    }
                )

        if not hydrated:
            print("❌  未找到相关文档，请先运行 ingest.py 摄取数据。")
            return 0

        # 6. Rerank
        final_items = hydrated
        rerank_fallback = False
        if not args.no_rerank:
            if args.verbose:
                _print_stage("⚖️ ", "Reranking (Cross-Encoder)")
            reranker = Reranker(settings)
            rerank_result = reranker.rerank(effective_query, hydrated, timeout_s=10.0, trace=trace)
            final_items = list(rerank_result.items or [])
            rerank_fallback = bool(rerank_result.fallback)
            if args.verbose:
                status = "⚠️  Fallback used" if rerank_fallback else "✅  Success"
                print(f"   - Status: {status}")

        # 7. Final Results
        _print_stage("🎯", f"Top {final_top_k} Results:")
        _print_ranked_items(final_items, top_k=final_top_k)

        trace.finish()
        write_trace(trace.to_dict(), settings=settings)

        return 0
    except Exception as e:
        _print_exception_chain(e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
