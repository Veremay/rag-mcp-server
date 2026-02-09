import logging
from typing import Any, List, Optional

from src.libs.reranker.base_reranker import BaseReranker

logger = logging.getLogger(__name__)


class CrossEncoderReranker(BaseReranker):
    """
    Reranker implementation that uses a Cross-Encoder model (e.g., sentence-transformers)
    to score query-candidate pairs and reorder candidates.
    """

    def __init__(self, model_name: str, scorer: Any = None):
        """
        Initialize the Cross-Encoder Reranker.

        Args:
            model_name: Name or path of the model to load (if scorer is not provided).
            scorer: Pre-initialized scorer instance (e.g. for testing or shared instance).
                    If None, attempts to load sentence_transformers.CrossEncoder.
        """
        self.model_name = model_name
        self.scorer = scorer

        if self.scorer is None:
            self._load_model()

    def _load_model(self):
        """Attempt to load the CrossEncoder model."""
        try:
            # Lazy import to avoid hard dependency if not used
            from sentence_transformers import CrossEncoder

            logger.info(f"Loading CrossEncoder model: {self.model_name}")
            self.scorer = CrossEncoder(self.model_name)
        except ImportError:
            logger.warning(
                "sentence-transformers library not installed. CrossEncoderReranker will fail if used."
            )
        except Exception as e:
            logger.error(f"Failed to load CrossEncoder model {self.model_name}: {e}")
            # We don't raise here to allow factory to create the instance,
            # but rerank() will fail if scorer is missing.

    def rerank(
        self,
        query: str,
        candidates: List[Any],
        top_k: Optional[int] = None,
        trace: Optional[Any] = None,
    ) -> List[Any]:
        """
        Rerank candidates using Cross-Encoder scores.

        Args:
            query: The search query.
            candidates: List of candidate objects.
            top_k: Number of results to return.
            trace: Trace context.

        Returns:
            Reordered list of candidates.

        Raises:
            RuntimeError: If scorer is not initialized.
            Exception: If prediction fails (propagated for fallback).
        """
        if not candidates:
            return []

        if self.scorer is None:
            raise RuntimeError(
                "CrossEncoder model is not initialized. Check logs for loading errors."
            )

        # Extract text content from candidates
        candidate_texts = []
        for cand in candidates:
            text = ""
            if hasattr(cand, "page_content"):
                text = cand.page_content
            elif hasattr(cand, "text"):
                text = cand.text
            elif isinstance(cand, dict) and "text" in cand:
                text = cand["text"]
            else:
                text = str(cand)
            candidate_texts.append(text)

        # Prepare pairs for scoring
        pairs = [[query, text] for text in candidate_texts]

        try:
            # Predict scores
            # Expecting scorer.predict to return a list/array of scores
            scores = self.scorer.predict(pairs)

            # Combine scores with candidates
            scored_candidates = list(zip(scores, candidates))

            # Sort by score descending
            scored_candidates.sort(key=lambda x: x[0], reverse=True)

            # Extract candidates
            ranked_results = [c for s, c in scored_candidates]

            if top_k:
                ranked_results = ranked_results[:top_k]

            return ranked_results

        except Exception as e:
            logger.error(f"CrossEncoder reranking failed: {e}")
            # Re-raise exception to signal failure to Core layer (triggering fallback)
            raise e
