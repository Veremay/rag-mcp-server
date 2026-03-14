"""
重排序模块：对检索得到的候选列表用 cross-encoder 等模型做精排。

粗排（稠密/稀疏）返回的 top_k 可能含噪声，用重排可把更相关的文档排到前面，
提升最终 top_m 的准确率。支持超时与 fallback：超时或异常时返回原序，避免整条检索链失败。
"""
from __future__ import annotations

import time
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
    """重排结果：items 为排序后的列表，fallback=True 表示未真正重排（超时/异常）沿用原序。"""
    items: List[Any]
    fallback: bool


class Reranker:
    """
    重排序器：在独立线程中调用 backend 重排，带超时与可观测性。

    用 ThreadPoolExecutor 做超时是为了避免 cross-encoder 卡住阻塞主流程；
    fallback 时返回原列表保证调用方总能拿到结果。合并时先重排结果再补原序剩余项，
    既保留重排带来的顺序又不会丢失未参与重排的候选。
    """

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
        """
        对 candidates 的前 top_m 条做重排，返回重排后的列表（含未参与重排的后续项）。

        top_m 限制参与重排的数量以控制延迟；超时或异常时 fallback 返回原序，
        并仍记录 record_stage 便于观测重排耗时与是否发生 fallback。
        """
        def record_stage(
            name: str,
            *,
            start_ms: float,
            end_ms: float,
            data: Optional[dict[str, Any]] = None,
            metrics: Optional[dict[str, float]] = None,
        ) -> None:
            if trace is None:
                return
            fn = getattr(trace, "record_stage", None)
            if not callable(fn):
                return
            fn(
                name,
                start_ms=float(start_ms),
                end_ms=float(end_ms),
                data=dict(data or {}),
                metrics=dict(metrics or {}),
            )

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
        start_ms = time.time() * 1000.0
        fallback = False
        reranked_head = []
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
            fallback = True
            return RerankResult(items=items, fallback=True)
        finally:
            end_ms = time.time() * 1000.0
            
            hits_to_log = reranked_head if not fallback else head
            
            record_stage(
                "rerank",
                start_ms=start_ms,
                end_ms=end_ms,
                data={
                    "top_m": effective_top_m,
                    "timeout_s": effective_timeout,
                    "n_candidates": len(items),
                    "hits": _serialize_rerank_hits(hits_to_log),
                },
                metrics={
                    "fallback": 1.0 if fallback else 0.0,
                    "n_input": float(len(head)),
                    "n_output": float(len(reranked_head))
                    if not fallback
                    else float(len(head)),
                },
            )

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
    """
    在子线程中执行重排并限制等待时间。不设 timeout 时直接调用，避免线程开销；
    设了 timeout 时用 ThreadPoolExecutor.result(timeout) 防止 backend 卡死。
    """
    if timeout_s is None:
        return list(fn(query, candidates, top_k=top_k, trace=trace))

    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn, query, candidates, top_k, trace)
        return list(fut.result(timeout=float(timeout_s)))


def _merge_preserving_recall(reranked: Sequence[T], original: Sequence[T]) -> List[T]:
    """
    合并重排后的前段与原始列表：先按重排顺序去重加入，再按原序补上未在重排结果中的项。
    用 id(x) 去重是为了保持对象引用一致，且不丢失「只在 original 里出现」的候选。
    """
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


TRACE_HITS_LIMIT = 20


def _serialize_rerank_hits(hits: Sequence[Any]) -> List[dict[str, Any]]:
    """
    将重排命中的多种形态（dict、HybridSearchHit、VectorRecord）统一序列化为 trace 用结构。
    限制条数避免 trace 体积过大；兼容多种类型是为了脚本与 MCP 等不同调用方都能被观测。
    """
    out = []
    for h in hits[:TRACE_HITS_LIMIT]:
        # Handle dictionary (from scripts/query.py)
        if isinstance(h, dict):
            item = {
                "id": str(h.get("chunk_id") or h.get("id") or ""),
                "score": float(h.get("score") or 0.0),
                "content": str(h.get("text") or h.get("content") or "")[:500],
                "metadata": h.get("metadata", {}),
            }
            out.append(item)
            continue

        # Try HybridSearchHit pattern (duck typing)
        if hasattr(h, "chunk_id") and hasattr(h, "score"):
            item = {
                "id": str(h.chunk_id),
                "score": float(h.score) if h.score is not None else 0.0,
            }
            if hasattr(h, "record") and h.record:
                if hasattr(h.record, "content"):
                    item["content"] = h.record.content[:500] if h.record.content else ""
                if hasattr(h.record, "metadata"):
                    item["metadata"] = h.record.metadata
            out.append(item)
        # Try VectorRecord pattern
        elif hasattr(h, "id") and hasattr(h, "content"):
            item = {
                "id": str(h.id),
                "content": h.content[:500] if h.content else "",
                "metadata": getattr(h, "metadata", {}),
            }
            out.append(item)
        else:
            # Fallback for unknown objects
            safe_repr = str(h)[:200]
            out.append({"repr": safe_repr})
    return out

