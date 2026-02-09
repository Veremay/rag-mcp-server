from typing import Any, List, Optional

from src.core.settings import Settings
from src.libs.llm.llm_factory import LLMFactory
from src.libs.reranker.base_reranker import BaseReranker
from src.libs.reranker.cross_encoder_reranker import CrossEncoderReranker
from src.libs.reranker.llm_reranker import LLMReranker


class NoneReranker(BaseReranker):
    """
    No-op reranker that returns candidates in their original order.
    Used as a fallback or when reranking is disabled.
    """

    def rerank(
        self,
        query: str,
        candidates: List[Any],
        top_k: Optional[int] = None,
        trace: Optional[Any] = None,
    ) -> List[Any]:
        """Return candidates as-is, optionally truncated to top_k."""
        if top_k is not None:
            return candidates[:top_k]
        return candidates


class RerankerFactory:
    """Factory for creating Reranker instances based on configuration."""

    @staticmethod
    def create(settings: Settings) -> BaseReranker:
        """
        Create a Reranker instance.

        Args:
            settings: Global settings object containing rerank configuration.

        Returns:
            An instance of BaseReranker.

        Raises:
            ValueError: If the configured backend is not supported.
        """
        backend = settings.rerank.backend.lower()

        if backend == "none":
            return NoneReranker()

        if backend == "llm":
            llm = LLMFactory.create(settings)
            return LLMReranker(llm)

        if backend == "cross-encoder" or backend == "cross_encoder":
            model_name = settings.rerank.model
            return CrossEncoderReranker(model_name=model_name)

        raise ValueError(f"Unsupported reranker backend: {backend}")
