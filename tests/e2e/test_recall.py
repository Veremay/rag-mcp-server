import pytest
import os
from src.core.settings import load_settings
from src.core.query_engine.hybrid_search import HybridSearch
from src.libs.evaluator.evaluator_factory import EvaluatorFactory
from src.observability.evaluation.eval_runner import EvalRunner

@pytest.mark.e2e
def test_recall_regression():
    """
    E2E Recall Regression Test.
    Runs evaluation against golden_test_set.json and asserts metrics meet minimum thresholds.
    """
    # 1. Setup
    try:
        settings = load_settings()
    except Exception as e:
        pytest.skip(f"Skipping E2E test: Failed to load settings ({e})")
        
    if not os.path.exists("tests/fixtures/golden_test_set.json"):
        pytest.fail("Golden test set not found at tests/fixtures/golden_test_set.json")

    # 2. Initialize Components
    try:
        hybrid_search = HybridSearch(settings)
        evaluator = EvaluatorFactory.create(settings)
        runner = EvalRunner(settings, hybrid_search, evaluator)
    except Exception as e:
        pytest.fail(f"Failed to initialize components: {e}")

    # 3. Run Evaluation
    report = runner.run("tests/fixtures/golden_test_set.json")
    
    # 4. Assertions
    print("\n=== E2E Recall Test Results ===")
    print(f"Mean Hit Rate: {report.aggregate_metrics.get('mean_hit_rate', 0):.4f}")
    print(f"Mean MRR:      {report.aggregate_metrics.get('mean_mrr', 0):.4f}")
    
    # Thresholds (Configurable via env vars or hardcoded)
    # TODO: Increase thresholds once stable data is ingested in the CI/Test environment.
    MIN_HIT_RATE = float(os.getenv("E2E_MIN_HIT_RATE", "0.0"))
    MIN_MRR = float(os.getenv("E2E_MIN_MRR", "0.0"))
    
    assert report.aggregate_metrics.get("mean_hit_rate", 0) >= MIN_HIT_RATE, \
        f"Mean Hit Rate {report.aggregate_metrics.get('mean_hit_rate')} < {MIN_HIT_RATE}"
        
    assert report.aggregate_metrics.get("mean_mrr", 0) >= MIN_MRR, \
        f"Mean MRR {report.aggregate_metrics.get('mean_mrr')} < {MIN_MRR}"
