import json
import time
from typing import Any, Dict, List, Optional

from src.core.query_engine.hybrid_search import HybridSearch
from src.core.settings import Settings
from src.libs.evaluator.base_evaluator import BaseEvaluator
from src.libs.llm.base_llm import BaseLLM


class EvalRunner:
    """
    Runs evaluation against a golden test set.
    Integrates Retrieval, Generation, and Evaluation.
    """

    def __init__(
        self,
        settings: Settings,
        evaluator: BaseEvaluator,
        retriever: Optional[HybridSearch] = None,
        llm: Optional[BaseLLM] = None,
    ):
        self.settings = settings
        self.evaluator = evaluator
        self.retriever = retriever
        self.llm = llm

    def load_golden_set(self, path: str) -> List[Dict[str, Any]]:
        """Load golden test set from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def run(
        self, golden_set_path: str, top_k: int = 3, verbose: bool = False
    ) -> Dict[str, float]:
        """
        Run the evaluation pipeline.

        Args:
            golden_set_path: Path to the golden test set JSON file.
            top_k: Number of documents to retrieve.
            verbose: Whether to print progress.

        Returns:
            Aggregated metrics (average across all queries).
        """
        data = self.load_golden_set(golden_set_path)
        all_metrics: Dict[str, List[float]] = {}
        results = []

        print(f"Starting evaluation on {len(data)} items...")

        for i, item in enumerate(data):
            query = item["query"]
            golden_answer = item["golden_answer"]
            golden_ids = item.get("golden_context_ids", [])

            if verbose:
                print(f"\nProcessing [{i+1}/{len(data)}]: {query}")

            # 1. Retrieval
            retrieved_texts = []
            retrieved_ids = []
            
            if self.retriever:
                hits = self.retriever.search(query, top_k_final=top_k)
                for hit in hits:
                    retrieved_ids.append(hit.chunk_id)
                    # Try to get text from record, fallback to empty string
                    text = hit.record.text if hit.record else ""
                    retrieved_texts.append(text)
            else:
                # Mock retrieval if no retriever provided
                retrieved_texts = ["Mock context 1", "Mock context 2"]
                retrieved_ids = ["mock_doc_1", "mock_doc_2"]

            # 2. Generation
            generated_answer = ""
            if self.llm:
                context_str = "\n\n".join(retrieved_texts)
                prompt = [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant. Answer the user query based on the provided context.",
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n{context_str}\n\nQuery: {query}\nAnswer:",
                    },
                ]
                generated_answer = self.llm.chat(prompt)
            else:
                generated_answer = "Mock generated answer."

            # 3. Evaluation
            eval_result = self.evaluator.evaluate(
                query=query,
                retrieved_ids=retrieved_ids,
                golden_ids=golden_ids,
                retrieved_texts=retrieved_texts,
                golden_answer=golden_answer,
                generated_answer=generated_answer,
            )

            if verbose:
                print(f"  Generated: {generated_answer[:50]}...")
                print(f"  Metrics: {eval_result}")

            results.append({
                "id": item.get("id", str(i)),
                "query": query,
                "metrics": eval_result
            })

            # Aggregate metrics
            for k, v in eval_result.items():
                if k not in all_metrics:
                    all_metrics[k] = []
                all_metrics[k].append(v)

        # Calculate averages
        avg_metrics = {}
        for k, v in all_metrics.items():
            if v:
                avg_metrics[k] = sum(v) / len(v)
            else:
                avg_metrics[k] = 0.0

        print("\nEvaluation Completed.")
        print(f"Average Metrics: {avg_metrics}")
        
        return avg_metrics
