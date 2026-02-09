from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from src.core.query_engine.hybrid_search import HybridSearch
from src.core.query_engine.reranker import Reranker
from src.core.response.response_builder import ResponseBuilder
from src.core.settings import Settings, load_settings

JsonDict = Dict[str, Any]


@dataclass(frozen=True)
class QueryKnowledgeHubParams:
    query: str
    top_k: Optional[int] = None
    collection: Optional[str] = None


def query_knowledge_hub(
    params: QueryKnowledgeHubParams, *, trace: Optional[Any] = None
) -> JsonDict:
    normalized_query = (params.query or "").strip()
    if not normalized_query:
        raise ValueError("query must be a non-empty string")

    settings = _get_settings()
    collection = (params.collection or "").strip() or None
    if not _has_any_data(settings, collection=collection):
        return ResponseBuilder().build([], query=normalized_query)

    query_for_search = (
        f"collection:{collection} {normalized_query}"
        if collection
        else normalized_query
    )

    top_k = params.top_k
    effective_top_k = int(top_k) if top_k is not None else None
    if effective_top_k is not None and effective_top_k <= 0:
        return ResponseBuilder().build([], query=normalized_query)

    searcher = HybridSearch(settings)
    hits = searcher.search(
        query_for_search,
        top_k_final=effective_top_k,
        trace=trace,
    )

    reranker = Reranker(settings)
    reranked = reranker.rerank(
        normalized_query,
        hits,
        top_m=None,
        trace=trace,
    )

    final_hits = reranked.items
    if effective_top_k is not None:
        final_hits = final_hits[:effective_top_k]

    return ResponseBuilder().build(final_hits, query=normalized_query)


@lru_cache(maxsize=1)
def _get_settings() -> Settings:
    return load_settings()


def _has_any_data(settings: Settings, *, collection: Optional[str]) -> bool:
    backend = str(
        getattr(getattr(settings, "vector_store", None), "backend", "")
    ).lower()
    persist_path = str(
        getattr(getattr(settings, "vector_store", None), "persist_path", "")
    )
    base_dir = Path(persist_path or ".")

    if backend == "chroma":
        if base_dir.exists():
            any_files = any(p.is_file() for p in base_dir.rglob("*"))
            if any_files:
                return True

    if backend == "jsonl":
        name = collection or str(getattr(settings.vector_store, "collection_name", ""))
        if name:
            jsonl_path = base_dir / f"{name}.jsonl"
            if jsonl_path.exists() and jsonl_path.stat().st_size > 0:
                return True

    bm25_dir = Path("data/db/bm25") / str(
        collection or getattr(settings.vector_store, "collection_name", "knowledge_hub")
    )
    if (bm25_dir / "meta.json").exists() and (bm25_dir / "postings.json").exists():
        return True

    return False
