from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, TypeVar

from src.core.settings import Settings
from src.libs.reranker.base_reranker import BaseReranker
from src.libs.reranker.reranker_factory import RerankerFactory

T = TypeVar("T")


@dataclass(frozen=True)
class RerankResult:
    items: List[Any]
    fallback: bool


class Reranker:
    def __init__(
        self,
        settings: Settings,
        *,
        backend: Optional[BaseReranker] = None,
        timeout_s: Optional[float] = None,
    ) -> None:
        self._settings = settings
        self._backend = backend or RerankerFactory.create(settings)
        self._timeout_s = timeout_s

    def rerank(
        self,
        query: str,
        candidates: Sequence[T],
        *,
        top_m: Optional[int] = None,
        timeout_s: Optional[float] = None,
        trace: Optional[Any] = None,
    ) -> RerankResult:
        normalized_query = (query or "").strip()
        if not normalized_query:
            return RerankResult(items=[], fallback=False)

        items = list(candidates)
        if not items:
            return RerankResult(items=[], fallback=False)

        effective_top_m = (
            int(top_m)
            if top_m is not None
            else int(getattr(self._settings.rerank, "top_m", len(items)))
        )
        if effective_top_m <= 0:
            return RerankResult(items=[], fallback=False)

        head = items[:effective_top_m]
        tail = items[effective_top_m:]

        effective_timeout = self._timeout_s if timeout_s is None else timeout_s
        try:
            reranked_head = _call_with_timeout(
                self._backend.rerank,
                effective_timeout,
                normalized_query,
                head,
                None,
                trace,
            )
        except (FutureTimeoutError, Exception):
            return RerankResult(items=items, fallback=True)

        merged_head = _merge_preserving_recall(reranked_head, head)
        return RerankResult(items=merged_head + tail, fallback=False)


def _call_with_timeout(
    fn: Any,
    timeout_s: Optional[float],
    query: str,
    candidates: List[Any],
    top_k: Optional[int],
    trace: Optional[Any],
) -> List[Any]:
    if timeout_s is None:
        return list(fn(query, candidates, top_k=top_k, trace=trace))

    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn, query, candidates, top_k, trace)
        return list(fut.result(timeout=float(timeout_s)))


def _merge_preserving_recall(reranked: Sequence[T], original: Sequence[T]) -> List[T]:
    seen = set()
    out: List[T] = []

    for x in reranked:
        key = id(x)
        if key in seen:
            continue
        seen.add(key)
        out.append(x)

    for x in original:
        key = id(x)
        if key in seen:
            continue
        seen.add(key)
        out.append(x)

    return out
