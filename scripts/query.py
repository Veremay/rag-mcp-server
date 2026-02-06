from __future__ import annotations

import argparse
import importlib.util
import sys
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
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--config", default="config/settings.yaml")
    return parser.parse_args(argv)


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _format_snippet(text: str, *, max_len: int = 160) -> str:
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


def _print_stage(title: str) -> None:
    print()
    print(f"== {title} ==")


def _print_ranked_items(items: List[Dict[str, Any]], *, top_k: int) -> None:
    for i, it in enumerate(items[:top_k], start=1):
        score = it.get("score")
        score_str = f"{float(score):.6f}" if isinstance(score, (int, float)) else "-"
        metadata = it.get("metadata")
        metadata_dict = metadata if isinstance(metadata, dict) else {}
        source = str(
            metadata_dict.get("source_path", "")
            or metadata_dict.get("source", "")
            or "-"
        )
        page = _extract_page(metadata_dict) or "-"
        text = str(it.get("text", "") or "")
        print(
            f"[{i:02d}] score={score_str} source={source} page={page} id={it.get('chunk_id','-')}\n"
            f"     {_format_snippet(text)}"
        )


def _print_dense_hits(hits: List[Any]) -> None:
    for i, h in enumerate(hits, start=1):
        record = getattr(h, "record", None)
        if record is None:
            continue
        chunk_id = getattr(record, "id", "-")
        metadata = getattr(record, "metadata", {}) or {}
        source = str(metadata.get("source_path", "") or "-")
        page = _extract_page(metadata) or "-"
        score = getattr(h, "score", None)
        score_str = f"{float(score):.6f}" if isinstance(score, (int, float)) else "-"
        content = getattr(record, "content", "") or ""
        print(
            f"[{i:02d}] score={score_str} source={source} page={page} id={chunk_id}\n"
            f"     {_format_snippet(str(content))}"
        )


def _print_sparse_hits(hits: List[Any]) -> None:
    for i, h in enumerate(hits, start=1):
        chunk_id = getattr(h, "chunk_id", "-")
        score = getattr(h, "score", None)
        score_str = f"{float(score):.6f}" if isinstance(score, (int, float)) else "-"
        print(f"[{i:02d}] score={score_str} id={chunk_id}")


def _print_fusion_hits(hits: List[Any]) -> None:
    for i, h in enumerate(hits, start=1):
        chunk_id = getattr(h, "chunk_id", "-")
        score = getattr(h, "score", None)
        score_str = f"{float(score):.6f}" if isinstance(score, (int, float)) else "-"
        dense_rank = getattr(h, "dense_rank", None)
        sparse_rank = getattr(h, "sparse_rank", None)
        print(
            f"[{i:02d}] score={score_str} id={chunk_id} dense_rank={dense_rank} sparse_rank={sparse_rank}"
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

    if str(settings.vector_store.backend).lower() == "chroma" and not _module_available(
        "chromadb"
    ):
        print(
            "ERROR: 当前配置 vector_store.backend=chroma，但环境中未安装 chromadb。",
            file=sys.stderr,
        )
        print(
            "  解决方案 1：安装依赖：pip install chromadb",
            file=sys.stderr,
        )
        print(
            "  解决方案 2：将 config/settings.yaml 中 vector_store.backend 改为 jsonl",
            file=sys.stderr,
        )
        return 1

    if (
        str(settings.embedding.provider).lower() == "openai"
        and not settings.embedding.api_key
    ):
        print(
            "ERROR: 当前配置 embedding.provider=openai，但未配置 embedding.api_key。",
            file=sys.stderr,
        )
        print(
            "  解决方案 1：在 config/settings.yaml 配置 embedding.api_key",
            file=sys.stderr,
        )
        print(
            "  解决方案 2：将 config/settings.yaml 中 embedding.provider 改为 local",
            file=sys.stderr,
        )
        return 1

    effective_query = query
    if args.collection and "collection:" not in effective_query:
        effective_query = f"collection:{args.collection} {effective_query}".strip()

    try:
        qp = QueryProcessor()
        processed = qp.process(effective_query)
        filters = dict(processed.filters or {})
        if args.collection:
            filters.setdefault("collection", str(args.collection))

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

        try:
            dense_hits = dense.retrieve(
                effective_query,
                filters=filters,
                top_k=dense_top_k if dense_top_k > 0 else None,
            )
        except RuntimeError as e:
            msg = str(e)
            if str(settings.embedding.provider).lower() == "openai" and (
                "Failed to connect to OpenAI Embedding API" in msg
                or "Request timed out" in msg
                or "timed out" in msg
            ):
                print(
                    "WARN: OpenAI Embedding 调用失败（疑似超时/网络问题），已自动回退到 local embedding（fake vectors）。",
                    file=sys.stderr,
                )
                print(
                    "      注意：该回退仅用于本地调试链路可用性，相关性可能明显下降。",
                    file=sys.stderr,
                )
                from src.libs.embedding.local_embedding import LocalEmbedding

                dense = DenseRetriever(
                    settings,
                    embedding=LocalEmbedding(
                        model=settings.embedding.model or "text-embedding-3-small",
                        dimension=1536,
                    ),
                    vector_store=getattr(dense, "_vector_store", None),
                )
                dense_hits = dense.retrieve(
                    effective_query,
                    filters=filters,
                    top_k=dense_top_k if dense_top_k > 0 else None,
                )
            else:
                raise
        sparse_hits = sparse.retrieve(
            sparse_query,
            filters=filters,
            top_k=sparse_top_k if sparse_top_k > 0 else None,
            collection=str(args.collection) if args.collection else None,
        )

        need_candidates = final_top_k
        if not args.no_rerank:
            need_candidates = max(
                need_candidates, int(getattr(settings.rerank, "top_m", need_candidates))
            )

        fused_hits = fusion.fuse(dense_hits, sparse_hits, top_k=need_candidates)

        dense_by_id = {
            h.record.id: h.record
            for h in dense_hits
            if getattr(h, "record", None) is not None
        }
        hydrated: List[Dict[str, Any]] = []
        for fh in fused_hits:
            chunk_id = getattr(fh, "chunk_id", None)
            if not isinstance(chunk_id, str):
                continue
            record = dense_by_id.get(
                chunk_id
            ) or _resolve_record_from_dense_vector_store(dense, chunk_id)
            if record is None:
                continue
            item = {
                "chunk_id": chunk_id,
                "text": record.content,
                "metadata": dict(record.metadata or {}),
                "score": float(getattr(fh, "score", 0.0)),
            }
            hydrated.append(item)

        if args.verbose:
            _print_stage("Dense")
            _print_dense_hits(dense_hits)
            _print_stage("Sparse")
            _print_sparse_hits(sparse_hits)
            _print_stage("Fusion")
            _print_fusion_hits(fused_hits)

        if not hydrated:
            print("未找到相关文档，请先运行 ingest.py 摄取数据。")
            return 0

        if args.no_rerank:
            final_items = hydrated[:final_top_k]
            rerank_fallback = True
        else:
            reranker = Reranker(settings)
            rerank_result = reranker.rerank(effective_query, hydrated, timeout_s=5.0)
            final_items = list(rerank_result.items or [])[:final_top_k]
            rerank_fallback = bool(rerank_result.fallback)

        if args.verbose:
            _print_stage(f"Rerank (fallback={str(rerank_fallback).lower()})")
            _print_ranked_items(final_items, top_k=final_top_k)

        if not args.verbose:
            _print_ranked_items(final_items, top_k=final_top_k)

        return 0
    except Exception as e:
        _print_exception_chain(e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
