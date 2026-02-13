import os
import json
import shutil
from pathlib import Path
from typing import Generator
import pytest
from streamlit.testing.v1 import AppTest

# Import the page render functions
# We need to ensure src is in python path, which pytest handles.
# However, AppTest runs in a separate context, so we might need to be careful with imports in the script string.

@pytest.fixture
def dashboard_env(tmp_path: Path) -> Generator[Path, None, None]:
    """
    Sets up a temporary environment for Dashboard testing.
    1. Sets env vars to point to tmp_path.
    2. Creates dummy data (traces, vector store).
    """
    # 1. Setup Env Vars
    # We use a subfolder for data to mimic the real structure
    data_root = tmp_path / "data"
    data_root.mkdir()
    
    logs_dir = data_root / "logs"
    logs_dir.mkdir()
    
    vector_store_dir = data_root / "vector_store"
    vector_store_dir.mkdir()
    
    # Copy config directory to data_root so Settings can find it
    repo_root = Path(os.getcwd())
    config_src = repo_root / "config"
    config_dst = data_root / "config"
    shutil.copytree(config_src, config_dst)

    # Save original env
    old_env = os.environ.copy()
    
    # Set new env vars
    # Point persist_path to the vector_store_dir
    os.environ["MODULAR_RAG_VECTOR_STORE__PERSIST_PATH"] = str(vector_store_dir)
    # Point logs_dir (if configurable) or ensure the code finds it.
    # The code usually assumes `logs/traces.jsonl` relative to CWD or configured path.
    # Let's check where TraceService looks for logs.
    # Assuming it uses a constant or setting. If it's hardcoded to "logs/traces.jsonl", we might need to run tests from data_root.
    # But AppTest runs the script.
    
    # For now, let's assume we can control it via env or it uses a relative path.
    # If it uses relative path "logs/traces.jsonl", we need to make sure the CWD is correct.
    # AppTest allows setting default_timeout, but CWD?
    
    # Let's inspect TraceService location logic later if it fails.
    # For now, we will create "logs" in the root of the repo (or current CWD during test).
    # Since we don't want to pollute repo, we rely on the fact that TraceService *should* be configurable.
    # If not, we might have to patch TraceService.
    
    # Create dummy traces.jsonl
    trace_file = logs_dir / "traces.jsonl"
    dummy_traces = [
        {
            "trace_id": "trace_001",
            "trace_type": "ingestion",  # Aligned with TraceService
            "operation": "ingest",
            "latency_ms": 100,
            "success": True,
            "started_ms": 1704103200000, # 2024-01-01T10:00:00
            "stages": [
                {"name": "load", "latency_ms": 10},
                {"name": "transform", "latency_ms": 20},
                {"name": "embed", "latency_ms": 50},
                {"name": "upsert", "latency_ms": 20}
            ],
            "metadata": {"file_count": 1, "chunk_count": 5}
        },
        {
            "trace_id": "trace_002",
            "trace_type": "query", # Aligned with TraceService
            "operation": "query",
            "latency_ms": 50,
            "success": True,
            "started_ms": 1704103500000, # 2024-01-01T10:05:00
            "query": "test query",
            "result_count": 2,
            "stages": [
                {"name": "rewrite", "latency_ms": 5},
                {"name": "retrieve", "latency_ms": 30},
                {"name": "rerank", "latency_ms": 15}
            ]
        }
    ]
    
    with open(trace_file, "w") as f:
        for t in dummy_traces:
            f.write(json.dumps(t) + "\n")
            
    # Create dummy vector store data (JsonlStore)
    # JsonlStore uses {collection_name}.jsonl in persist_path
    # Default collection is likely "default" or "modular_rag"
    collection_name = "test_collection"
    os.environ["MODULAR_RAG_VECTOR_STORE__COLLECTION_NAME"] = collection_name
    
    vector_file = vector_store_dir / f"{collection_name}.jsonl"
    dummy_vectors = [
        {
            "id": "chunk_001",
            "text": "This is a test document.",
            "metadata": {"source": "test.pdf", "page": 1},
            "embedding": [0.1] * 384 # minimal embedding
        },
        {
            "id": "chunk_002",
            "text": "Another test chunk.",
            "metadata": {"source": "test.pdf", "page": 2},
            "embedding": [0.2] * 384
        }
    ]
    
    with open(vector_file, "w") as f:
        for v in dummy_vectors:
            f.write(json.dumps(v) + "\n")

    yield data_root

    # Cleanup
    os.environ.clear()
    os.environ.update(old_env)

# Helper to run a page function
def run_page_test(page_import_path: str, function_name: str, tmp_path: Path, cwd: Path):
    """
    Creates a runner script and executes it with AppTest.
    """
    script_content = f"""
import os
import sys
from pathlib import Path

# Set CWD to the data root so "logs/traces.jsonl" works
os.chdir("{cwd}")

# Add src to path (assuming original CWD was repo root)
# We need to find where 'src' is.
# Since we passed os.getcwd() from the test runner (which is repo root),
# we can just add that.
sys.path.append("{os.getcwd()}")

from {page_import_path} import {function_name}

# Execute the render function
{function_name}()
"""
    runner_path = tmp_path / f"runner_{function_name}.py"
    runner_path.write_text(script_content)
    
    # Run AppTest
    at = AppTest.from_file(str(runner_path), default_timeout=30)
    at.run()
    
    return at

@pytest.mark.e2e
def test_dashboard_overview_smoke(dashboard_env, tmp_path):
    """Smoke test for Overview page."""
    at = run_page_test(
        "src.observability.dashboard.pages.overview",
        "render_overview",
        tmp_path,
        dashboard_env
    )
    assert not at.exception, f"Overview page failed: {at.exception}"

@pytest.mark.e2e
def test_dashboard_data_browser_smoke(dashboard_env, tmp_path):
    """Smoke test for Data Browser page."""
    at = run_page_test(
        "src.observability.dashboard.pages.data_browser",
        "render_data_browser_page",
        tmp_path,
        dashboard_env
    )
    assert not at.exception, f"Data Browser page failed: {at.exception}"

@pytest.mark.e2e
def test_dashboard_ingestion_traces_smoke(dashboard_env, tmp_path):
    """Smoke test for Ingestion Traces page."""
    at = run_page_test(
        "src.observability.dashboard.pages.ingestion_traces",
        "render_ingestion_traces_page",
        tmp_path,
        dashboard_env
    )
    assert not at.exception, f"Ingestion Traces page failed: {at.exception}"

@pytest.mark.e2e
def test_dashboard_query_traces_smoke(dashboard_env, tmp_path):
    """Smoke test for Query Traces page."""
    at = run_page_test(
        "src.observability.dashboard.pages.query_traces",
        "render_query_traces_page",
        tmp_path,
        dashboard_env
    )
    assert not at.exception, f"Query Traces page failed: {at.exception}"

@pytest.mark.e2e
def test_dashboard_ingestion_manager_smoke(dashboard_env, tmp_path):
    """Smoke test for Ingestion Manager page."""
    at = run_page_test(
        "src.observability.dashboard.pages.ingestion_manager",
        "render_ingestion_manager_page",
        tmp_path,
        dashboard_env
    )
    assert not at.exception, f"Ingestion Manager page failed: {at.exception}"

@pytest.mark.e2e
def test_dashboard_evaluation_panel_smoke(dashboard_env, tmp_path):
    """Smoke test for Evaluation Panel page."""
    at = run_page_test(
        "src.observability.dashboard.pages.evaluation_panel",
        "render_evaluation_panel",
        tmp_path,
        dashboard_env
    )
    assert not at.exception, f"Evaluation Panel page failed: {at.exception}"
