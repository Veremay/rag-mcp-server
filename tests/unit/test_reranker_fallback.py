import time
from typing import Any, List, Optional
from unittest.mock import MagicMock

from src.core.query_engine.reranker import Reranker
from src.core.settings import Settings
from src.libs.reranker.base_reranker import BaseReranker


class ExplodingBackend(BaseReranker):
    def rerank(
        self,
        query: str,
        candidates: List[Any],
        top_k: Optional[int] = None,
        trace: Optional[Any] = None,
    ) -> List[Any]:
        raise RuntimeError("boom")


class SleepingBackend(BaseReranker):
    def __init__(self, sleep_s: float) -> None:
        self._sleep_s = float(sleep_s)

    def rerank(
        self,
        query: str,
        candidates: List[Any],
        top_k: Optional[int] = None,
        trace: Optional[Any] = None,
    ) -> List[Any]:
        time.sleep(self._sleep_s)
        return candidates


class ReverseBackend(BaseReranker):
    def rerank(
        self,
        query: str,
        candidates: List[Any],
        top_k: Optional[int] = None,
        trace: Optional[Any] = None,
    ) -> List[Any]:
        return list(reversed(candidates))


def _settings(top_m: int = 3) -> Settings:
    settings = MagicMock(spec=Settings)
    settings.rerank = MagicMock()
    settings.rerank.top_m = int(top_m)
    return settings


def test_reranker_falls_back_on_backend_exception() -> None:
    candidates = ["a", "b", "c", "d"]
    r = Reranker(_settings(top_m=3), backend=ExplodingBackend())

    out = r.rerank("q", candidates)
    assert out.fallback is True
    assert out.items == candidates


def test_reranker_falls_back_on_timeout() -> None:
    candidates = ["a", "b", "c", "d"]
    r = Reranker(_settings(top_m=3), backend=SleepingBackend(0.05))

    out = r.rerank("q", candidates, timeout_s=0.01)
    assert out.fallback is True
    assert out.items == candidates


def test_reranker_reranks_top_m_and_preserves_tail_on_success() -> None:
    candidates = ["a", "b", "c", "d"]
    r = Reranker(_settings(top_m=3), backend=ReverseBackend())

    out = r.rerank("q", candidates)
    assert out.fallback is False
    assert out.items == ["c", "b", "a", "d"]
