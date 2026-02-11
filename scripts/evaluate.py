#!/usr/bin/env python3
"""
Evaluation Script for Modular RAG.

This script runs the evaluation pipeline using the EvalRunner.
It loads a golden test set, performs retrieval and generation (or uses mocks),
and computes evaluation metrics.

Usage:
    python scripts/evaluate.py [--golden_set path/to/json]
"""

import argparse
import logging
import os
import sys

# Ensure src module is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.settings import load_settings
from src.libs.evaluator.evaluator_factory import EvaluatorFactory
from src.observability.evaluation.eval_runner import EvalRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run RAG Evaluation")
    parser.add_argument(
        "--golden_set",
        type=str,
        default="tests/fixtures/golden_test_set.json",
        help="Path to the golden test set JSON file"
    )
    args = parser.parse_args()

    try:
        # Load settings
        settings = load_settings()
        logger.info("Settings loaded successfully.")
        
        # Create evaluator
        evaluator = EvaluatorFactory.create(settings)
        logger.info(f"Evaluator created: {evaluator.__class__.__name__}")
        
        # Initialize EvalRunner
        # Note: We are passing None for retriever and llm for now to demonstrate
        # the runner's capability without requiring a full RAG stack setup.
        # In a real scenario, you would instantiate HybridSearch and an LLM here.
        runner = EvalRunner(
            settings=settings,
            evaluator=evaluator,
            retriever=None,  # TODO: Instantiate HybridSearch(settings)
            llm=None         # TODO: Instantiate LLMFactory.create(settings)
        )
        
        # Run evaluation
        logger.info(f"Running evaluation on {args.golden_set}...")
        results = runner.run(args.golden_set, verbose=True)
        
        print("\n" + "="*40)
        print(" FINAL AGGREGATED METRICS ")
        print("="*40)
        if results:
            for metric, score in results.items():
                print(f"{metric}: {score:.4f}")
        else:
            print("No metrics computed.")
        print("="*40 + "\n")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
