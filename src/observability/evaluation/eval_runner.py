import json
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from pathlib import Path

from src.core.settings import Settings
from src.core.query_engine.hybrid_search import HybridSearch
from src.libs.evaluator.base_evaluator import BaseEvaluator


@dataclass
class EvalCaseResult:
    query: str
    metrics: Dict[str, float]
    retrieved_ids: List[str]
    retrieved_sources: List[str]
    golden_ids: List[str]
    golden_sources: List[str]


@dataclass
class EvalReport:
    total_cases: int
    aggregate_metrics: Dict[str, float]
    case_results: List[EvalCaseResult] = field(default_factory=list)


class EvalRunner:
    """
    Runs evaluation against a golden test set using HybridSearch and an Evaluator.
    """

    def __init__(
        self,
        settings: Settings,
        hybrid_search: HybridSearch,
        evaluator: BaseEvaluator,
    ):
        self.settings = settings
        self.hybrid_search = hybrid_search
        self.evaluator = evaluator

    def run(self, test_set_path: str) -> EvalReport:
        """
        Execute evaluation suite.

        Args:
            test_set_path: Path to the golden test set JSON file.

        Returns:
            EvalReport containing detailed and aggregated results.
        """
        path = Path(test_set_path)
        if not path.exists():
            raise FileNotFoundError(f"Test set not found at: {test_set_path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        test_cases = data.get("test_cases", [])
        results: List[EvalCaseResult] = []
        aggregates: Dict[str, float] = {}

        for case in test_cases:
            query = case.get("query", "")
            golden_ids = case.get("expected_chunk_ids", [])
            golden_sources = case.get("expected_sources", [])

            # Run retrieval
            hits = self.hybrid_search.search(
                query,
                top_k_final=self.settings.retrieval.top_k_final,
            )

            retrieved_ids = [hit.chunk_id for hit in hits]
            retrieved_texts = [hit.record.content for hit in hits]
            retrieved_sources = [
                str(
                    hit.record.metadata.get("source_path")
                    or hit.record.metadata.get("source")
                    or ""
                )
                for hit in hits
            ]

            # Run evaluation
            metrics = self.evaluator.evaluate(
                query=query,
                retrieved_ids=retrieved_ids,
                golden_ids=golden_ids,
                retrieved_texts=retrieved_texts,
                golden_sources=golden_sources,
                retrieved_sources=retrieved_sources,
            )

            results.append(
                EvalCaseResult(
                    query=query,
                    metrics=metrics,
                    retrieved_ids=retrieved_ids,
                    retrieved_sources=retrieved_sources,
                    golden_ids=golden_ids,
                    golden_sources=golden_sources,
                )
            )

        # Calculate aggregates
        if results:
            metric_keys = results[0].metrics.keys()
            for key in metric_keys:
                values = [r.metrics.get(key, 0.0) for r in results]
                aggregates[f"mean_{key}"] = sum(values) / len(values)

        return EvalReport(
            total_cases=len(results),
            aggregate_metrics=aggregates,
            case_results=results,
        )
