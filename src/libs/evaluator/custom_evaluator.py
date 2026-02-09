from typing import Any, Dict, List, Optional

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
    ) -> Dict[str, float]:
        """
        Calculate Hit Rate and Mean Reciprocal Rank (MRR).

        Metrics:
        - hit_rate: 1.0 if at least one golden_id is in retrieved_ids, else 0.0.
        - mrr: 1 / (rank + 1) of the first relevant document found. 0.0 if none found.
        """
        if not golden_ids:
            return {"hit_rate": 0.0, "mrr": 0.0}

        golden_set = set(golden_ids)

        # Hit Rate
        hit = any(rid in golden_set for rid in retrieved_ids)
        hit_rate = 1.0 if hit else 0.0

        # MRR
        mrr = 0.0
        for i, rid in enumerate(retrieved_ids):
            if rid in golden_set:
                mrr = 1.0 / (i + 1)
                break

        return {"hit_rate": hit_rate, "mrr": mrr}
