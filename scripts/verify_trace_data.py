import sys
import os
import json
import logging
from unittest.mock import MagicMock
from pathlib import Path

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.settings import load_settings
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.models import Document
from src.libs.loader.base_loader import BaseLoader
from src.core.trace.trace_context import TraceContext

# Mock Loader
class MockLoader(BaseLoader):
    def load(self, file_path):
        # Generate enough text for multiple chunks
        text = "This is a sentence. " * 500  # 10000 chars
        return Document(
            text=text,
            metadata={"source": "mock_file.pdf"}
        )

def main():
    print("--- Verifying Trace Data Recording ---")
    
    # 1. Load Settings
    settings = load_settings()
    
    # Enable observability and set log file
    settings.observability.enabled = True
    settings.observability.log_file = "logs/traces_test.jsonl"
    
    # 2. Initialize Pipeline
    # Disable heavy transforms for test speed
    settings.ingestion.transform.chunk_refiner.enabled = False
    settings.ingestion.transform.metadata_enricher.enabled = False
    settings.ingestion.transform.image_captioner.enabled = False
    
    pipeline = IngestionPipeline(settings)
    
    # 3. Mock Loader Resolution
    pipeline._resolve_loader = MagicMock(return_value=MockLoader())
    
    # 4. Run Ingestion
    print("Running ingestion...", flush=True)
    try:
        result = pipeline.ingest(
            collection="test_collection",
            file_path="mock_file.pdf",
            force=True
        )
    except Exception as e:
        print(f"Ingestion FAILED with error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return
    
    print(f"Ingestion finished. Chunks: {len(result.chunks)}", flush=True)
    
    # 5. Verify Trace Log
    log_path = Path("logs/traces_test.jsonl")
    if not log_path.exists():
        print("[FAIL] Trace log file not created.")
        return
        
    print(f"Reading trace log: {log_path}")
    last_trace = None
    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        if lines:
            last_trace = json.loads(lines[-1])
            
    if not last_trace:
        print("[FAIL] No trace found in log.")
        return
        
    # 6. Check for Data Previews
    stages = {s["name"]: s for s in last_trace.get("stages", [])}
    
    # Check Split
    if "split" in stages:
        data = stages["split"].get("data", {})
        if "chunks_preview" in data:
            print("[PASS] Split stage has chunks_preview.")
            print(f"      Preview count: {len(data['chunks_preview'])}")
        else:
            print("[FAIL] Split stage MISSING chunks_preview.")
    else:
        print("[FAIL] Split stage not found.")
        
    # Check Transform
    if "transform" in stages:
        data = stages["transform"].get("data", {})
        if "chunks_preview" in data:
            print("[PASS] Transform stage has chunks_preview.")
        else:
            print("[FAIL] Transform stage MISSING chunks_preview.")
    else:
        print("[FAIL] Transform stage not found.")

    # Check Encode
    if "encode" in stages:
        data = stages["encode"].get("data", {})
        if "embedding_dim" in data:
             print(f"[PASS] Encode stage has embedding_dim: {data['embedding_dim']}")
        else:
             print("[FAIL] Encode stage MISSING embedding_dim.")

    # Check Upsert
    if "upsert" in stages:
        data = stages["upsert"].get("data", {})
        if "upserted_ids" in data:
             print(f"[PASS] Upsert stage has upserted_ids. Count: {len(data['upserted_ids'])}")
        else:
             print("[FAIL] Upsert stage MISSING upserted_ids.")
             
    # Cleanup
    if log_path.exists():
        os.remove(log_path)
    print("Test Complete.")

if __name__ == "__main__":
    main()
