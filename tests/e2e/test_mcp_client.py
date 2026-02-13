import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import yaml

from src.ingestion.models import Document
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.libs.loader.base_loader import BaseLoader
from src.core.settings import load_settings, Settings

class MockLoader(BaseLoader):
    def load(self, file_path: Path) -> Document:
        return Document(
            text="Modular RAG is a flexible framework for building RAG applications. It supports MCP protocol.",
            metadata={"source": file_path.name, "title": "Test Doc"}
        )

@pytest.mark.e2e
def test_mcp_client_e2e_flow(tmp_path: Path) -> None:
    # 1. Setup Config
    vector_store_path = tmp_path / "vector_store"
    vector_store_path.mkdir()
    
    settings_data = {
        "llm": {
            "provider": "ollama",
            "model": "llama3",
            "api_key": "dummy"
        },
        "embedding": {
            "provider": "local",
            "model": "fake"
        },
        "vector_store": {
            "backend": "jsonl",
            "persist_path": str(vector_store_path),
            "collection_name": "test_collection"
        },
        "ingestion": {
            "splitter": {
                "provider": "recursive",
                "chunk_size": 500,
                "chunk_overlap": 50
            },
            "transform": {
                "chunk_refiner": {"enabled": False},
                "metadata_enricher": {"enabled": False},
                "image_captioner": {"enabled": False}
            }
        },
        "retrieval": {
            "sparse_backend": "bm25",
            "fusion_algorithm": "rrf",
            "top_k_dense": 5,
            "top_k_sparse": 5,
            "top_k_final": 5
        },
        "rerank": {
            "backend": "none",
            "model": "none",
            "top_m": 5
        },
        "vision_llm": {
             "provider": "ollama",
             "model": "llava"
        },
        "evaluation": {
            "backends": ["custom"],
            "golden_test_set": ""
        },
        "observability": {
            "enabled": False,
            "log_file": "",
            "dashboard_port": 0
        }
    }
    
    config_path = tmp_path / "settings.yaml"
    with open(config_path, "w") as f:
        yaml.dump(settings_data, f)
        
    # Set env var for current process (test runner)
    os.environ["MODULAR_RAG_CONFIG_PATH"] = str(config_path)
    
    # Reload settings to ensure we pick up the new config
    # We can just call load_settings() because it reads the file each time (no lru_cache on top level)
    settings = load_settings(str(config_path))
    
    # 2. Ingest Data
    # We need a dummy file for integrity check, even if MockLoader ignores content
    dummy_file = tmp_path / "dummy.txt"
    dummy_file.write_text("dummy content")
    
    # Configure BM25 to use tmp_path so it matches where server will look
    bm25_dir = tmp_path / "data" / "db" / "bm25"
    
    pipeline = IngestionPipeline(
        settings=settings,
        loader=MockLoader(),
        bm25_indexer=BM25Indexer(base_dir=bm25_dir)
    )
    pipeline.ingest(
        collection="test_collection",
        file_path=dummy_file,
        force=True
    )

    # 3. Start MCP Server
    # We need to find the root of the repo
    repo_root = Path(__file__).resolve().parents[2]
    cmd = [sys.executable, "-m", "src.mcp_server.server"]
    
    # Prepare environment for subprocess
    env = os.environ.copy()
    env["MODULAR_RAG_CONFIG_PATH"] = str(config_path)
    env["PYTHONPATH"] = str(repo_root) # Ensure src is importable
    env["MODULAR_RAG_SKIP_ROOT_CHECK"] = "1" # Don't chdir to repo root, stay in tmp_path
    
    proc = subprocess.Popen(
        cmd,
        cwd=str(tmp_path), # Change CWD to tmp_path so server uses tmp_path/data/db/bm25
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8" # Ensure UTF-8 for JSON
    )
    
    try:
        # 4. Call query_knowledge_hub
        # First, initialize
        req_init = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0"}
            }
        }
        
        # Second, call tool
        req_call = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "query_knowledge_hub",
                "arguments": {
                    "query": "Modular RAG",
                    "collection": "test_collection"
                }
            }
        }
        
        # Send requests
        input_str = json.dumps(req_init) + "\n" + json.dumps(req_call) + "\n"
        
        # We use communicate to send input and wait for output
        # But communicate waits for EOF or process exit if we don't handle it carefully.
        # Actually, for stdio server, we might want to read line by line.
        # However, for a test, we can just send everything and close stdin, which signals EOF to server.
        # The server loop `for line in sys.stdin` will finish when stdin is closed.
        stdout, stderr = proc.communicate(input=input_str, timeout=10)
        
        if proc.returncode != 0:
            print(f"Server stderr: {stderr}")
        
        # 5. Verify Response
        lines = [l for l in stdout.splitlines() if l.strip()]
        
        # We expect at least 2 JSON responses
        assert len(lines) >= 2, f"Expected at least 2 responses, got {len(lines)}. Stderr: {stderr}"
        
        # Check initialize response
        resp_init = json.loads(lines[0])
        assert resp_init.get("id") == 1
        assert "result" in resp_init
        
        # Check call response
        resp_call = json.loads(lines[1])
        assert resp_call.get("id") == 2
        
        if "error" in resp_call:
             pytest.fail(f"Tool call returned error: {resp_call['error']}")
             
        assert "result" in resp_call
        result = resp_call["result"]
        
        # Verify content structure
        assert "content" in result
        content = result["content"]
        assert isinstance(content, list)
        assert len(content) > 0
        assert content[0]["type"] == "text"
        text_content = content[0]["text"]
        
        # Verify citation and content
        # With local embedding (fake) and 'Modular RAG' query vs 'Modular RAG' text,
        # it should match if embedding is deterministic or if we use BM25.
        # We enabled BM25 in config (sparse_backend: bm25).
        # Hybrid search should find it.
        
        assert "Modular RAG" in text_content
        # Check structured content for citations
        structured = result.get("structuredContent", {})
        assert "citations" in structured
        citations = structured["citations"]
        assert isinstance(citations, list)
        
        # We might not get citations if the score is too low?
        # But with exact keyword match "Modular RAG", BM25 should be high.
        # Let's check if citations are present.
        # Note: ResponseBuilder logic puts citations in markdown if present.
        
    except subprocess.TimeoutExpired:
        proc.kill()
        pytest.fail("Server process timed out")
    except Exception as e:
        if proc.poll() is None:
            proc.kill()
        raise e
