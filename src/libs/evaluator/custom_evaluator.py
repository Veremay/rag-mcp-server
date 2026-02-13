from typing import Any, Dict, List, Optional
from pathlib import Path

from src.libs.evaluator.base_evaluator import BaseEvaluator


class CustomEvaluator(BaseEvaluator):
    """
    Simple custom evaluator implementing Hit Rate and MRR.
    Does not require external API calls.
    """

    def evaluate(
        self,
        query: str,
        retrieved_ids: List[str],
        golden_ids: List[str],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """
        Calculate Hit Rate and Mean Reciprocal Rank (MRR).

        Metrics:
        - hit_rate: 1.0 if at least one golden_id is in retrieved_ids, else 0.0.
        - mrr: 1 / (rank + 1) of the first relevant document found. 0.0 if none found.
        """
        golden_sources = kwargs.get("golden_sources", [])
        retrieved_sources = kwargs.get("retrieved_sources", [])

        if not golden_ids and not golden_sources:
            return {"hit_rate": 0.0, "mrr": 0.0}

        golden_set = set(golden_ids)
        
        # Determine matches based on IDs or Sources
        matches = []
        for i, rid in enumerate(retrieved_ids):
            is_match = False
            # Check ID match
            if rid in golden_set:
                is_match = True
            
            # Check Source match if ID match failed and we have sources
            if not is_match and golden_sources and i < len(retrieved_sources):
                r_source = retrieved_sources[i]
                # Normalize paths for comparison (handle absolute vs relative)
                # We check if golden_source is a suffix of retrieved_source
                # e.g. "docs/foo.pdf" matches "/abs/path/to/docs/foo.pdf"
                for g_source in golden_sources:
                    if r_source.endswith(g_source):
                        is_match = True
                        break
            
            matches.append(is_match)

        # Hit Rate
        hit = any(matches)
        hit_rate = 1.0 if hit else 0.0

        # MRR
        mrr = 0.0
        for i, is_match in enumerate(matches):
            if is_match:
                mrr = 1.0 / (i + 1)
                break

        return {"hit_rate": hit_rate, "mrr": mrr}
