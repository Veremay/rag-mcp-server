import sys
import os
import logging
from dataclasses import dataclass

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.settings import load_settings
from src.ingestion.models import Chunk
from src.ingestion.transform.chunk_refiner import ChunkRefiner
from src.ingestion.transform.metadata_enricher import MetadataEnricher
# from src.ingestion.transform.image_captioner import ImageCaptioner # Requires actual image path or mock

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_chunk_refiner(settings):
    print("\n--- Testing Chunk Refiner ---")
    refiner = ChunkRefiner(settings)
    
    # Simulate a chunk with some noise (e.g., broken sentence from PDF)
    raw_text = "This is a sentence that is bro-\nken across lines. It also has page number 12 at the end."
    chunk = Chunk(
        id="test_chunk_1",
        text=raw_text,
        metadata={"source": "test_doc"},
        doc_id="doc_1",
        offset=0
    )
    
    print(f"Original Text:\n{chunk.text}")
    
    try:
        results = refiner.transform([chunk])
        refined_chunk = results[0]
        print(f"\nRefined Text:\n{refined_chunk.text}")
        
        if refined_chunk.text != raw_text and "bro-ken" not in refined_chunk.text:
             print("[PASS] Chunk Refiner modified the text.")
        else:
             print("[WARN] Chunk Refiner might not have changed the text significantly (or LLM returned same text).")
             
    except Exception as e:
        print(f"[FAIL] Chunk Refiner failed: {e}")

def test_metadata_enricher(settings):
    print("\n--- Testing Metadata Enricher ---")
    enricher = MetadataEnricher(settings)
    
    # Simulate a chunk that needs metadata
    chunk_text = """
    Python is a high-level, general-purpose programming language. 
    Its design philosophy emphasizes code readability with the use of significant indentation.
    Python is dynamically typed and garbage-collected.
    """
    chunk = Chunk(
        id="test_chunk_2",
        text=chunk_text,
        metadata={"source": "python_intro.txt"},
        doc_id="doc_2",
        offset=0
    )
    
    try:
        results = enricher.transform([chunk])
        enriched_chunk = results[0]
        print(f"\nGenerated Metadata:\n{enriched_chunk.metadata}")
        
        meta = enriched_chunk.metadata
        if "title" in meta and "summary" in meta and "tags" in meta:
             print("[PASS] Metadata Enricher generated required fields.")
        else:
             print(f"[FAIL] Missing metadata fields. Got: {meta.keys()}")
             
    except Exception as e:
        print(f"[FAIL] Metadata Enricher failed: {e}")

def main():
    try:
        settings = load_settings()
        print("Settings loaded successfully.")
        
        # Verify settings are actually enabled
        print(f"Chunk Refiner Enabled: {settings.ingestion.transform.chunk_refiner.enabled}")
        print(f"Chunk Refiner LLM: {settings.ingestion.transform.chunk_refiner.enable_llm}")
        print(f"Metadata Enricher Enabled: {settings.ingestion.transform.metadata_enricher.enabled}")
        print(f"Metadata Enricher LLM: {settings.ingestion.transform.metadata_enricher.enable_llm}")
        
        test_chunk_refiner(settings)
        test_metadata_enricher(settings)
        
    except Exception as e:
        print(f"Global Error: {e}")

if __name__ == "__main__":
    main()
