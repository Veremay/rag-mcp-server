import subprocess
import sys
from pathlib import Path

import pytest


def _write_config(path: Path) -> None:
    content = """
llm:
  provider: openai
  model: gpt-4o
  api_key: ""

embedding:
  provider: local
  model: local

vision_llm:
  provider: azure
  model: gpt-4o

vector_store:
  backend: jsonl
  persist_path: ./data/db/vector_store
  collection_name: knowledge_hub

ingestion:
  splitter:
    provider: recursive
    chunk_size: 1000
    chunk_overlap: 200

retrieval:
  sparse_backend: bm25
  fusion_algorithm: rrf
  top_k_dense: 20
  top_k_sparse: 20
  top_k_final: 10

rerank:
  backend: none
  model: none
  top_m: 30

evaluation:
  backends: [custom]
  golden_test_set: ./tests/fixtures/golden_test_set.json

observability:
  enabled: false
  log_file: ./logs/traces.jsonl
  dashboard_port: 8501
""".lstrip()
    path.write_text(content, encoding="utf-8")


@pytest.mark.e2e
def test_ingest_script_creates_artifacts_and_skips_incremental(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "ingest.py"

    config_path = tmp_path / "settings.yaml"
    _write_config(config_path)

    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 dummy")

    cmd = [
        sys.executable,
        str(script_path),
        "--collection",
        "c15",
        "--path",
        str(pdf_path),
        "--config",
        str(config_path),
    ]

    r1 = subprocess.run(cmd, cwd=str(tmp_path), capture_output=True, text=True)
    assert r1.returncode == 0
    assert "INGESTED" in r1.stdout

    meta_path = tmp_path / "data" / "db" / "bm25" / "c15" / "meta.json"
    postings_path = tmp_path / "data" / "db" / "bm25" / "c15" / "postings.json"
    history_path = tmp_path / "data" / "cache" / "ingestion_history.json"
    vector_path = tmp_path / "data" / "db" / "vector_store" / "c15.jsonl"
    assert meta_path.exists()
    assert postings_path.exists()
    assert history_path.exists()
    assert vector_path.exists()

    before_meta = meta_path.read_bytes()
    before_postings = postings_path.read_bytes()
    before_vector = vector_path.read_bytes()

    r2 = subprocess.run(cmd, cwd=str(tmp_path), capture_output=True, text=True)
    assert r2.returncode == 0
    assert "SKIPPED" in r2.stdout
    assert meta_path.read_bytes() == before_meta
    assert postings_path.read_bytes() == before_postings
    assert vector_path.read_bytes() == before_vector

    r3 = subprocess.run(
        cmd + ["--force"], cwd=str(tmp_path), capture_output=True, text=True
    )
    assert r3.returncode == 0
    assert "INGESTED" in r3.stdout
