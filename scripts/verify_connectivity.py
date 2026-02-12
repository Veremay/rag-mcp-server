
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.settings import load_settings
from src.libs.llm.llm_factory import LLMFactory
from src.libs.embedding.embedding_factory import EmbeddingFactory
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def verify_connectivity():
    print("="*50)
    print("Starting Configuration Verification")
    print("="*50)

    # 1. Load Settings
    try:
        settings = load_settings()
        print("[OK] Settings loaded successfully.")
        print(f"    LLM Provider: {settings.llm.provider}")
        print(f"    LLM Model: {settings.llm.model}")
        print(f"    Embedding Provider: {settings.embedding.provider}")
        print(f"    Embedding Model: {settings.embedding.model}")
        print(f"    Vision Provider: {settings.vision_llm.provider}")
        print(f"    Vision Model: {settings.vision_llm.model}")
    except Exception as e:
        print(f"[FAIL] Failed to load settings: {e}")
        return

    # 2. Test Main LLM
    print("-" * 30)
    print("Testing Main LLM Connectivity...")
    try:
        llm = LLMFactory.create(settings)
        response = llm.chat([{"role": "user", "content": "Hello, reply with 'OK'."}])
        print(f"[OK] Main LLM Check Passed. Response: {response}")
    except Exception as e:
        print(f"[FAIL] Main LLM Check Failed: {e}")

    # 3. Test Embedding
    print("-" * 30)
    print("Testing Embedding Connectivity...")
    try:
        embedder = EmbeddingFactory.create(settings)
        # BaseEmbedding.embed takes a list of strings
        vectors = embedder.embed(["test query"])
        print(f"[OK] Embedding Check Passed. Vector length: {len(vectors[0])}")
    except Exception as e:
        print(f"[FAIL] Embedding Check Failed: {e}")

    # 4. Test Vision LLM
    print("-" * 30)
    print("Testing Vision LLM Connectivity...")
    try:
        vision_llm = LLMFactory.create_vision(settings)
        # Just test simple chat capability first, as we might not have an image handy
        response = vision_llm.chat([{"role": "user", "content": "Hello, reply with 'OK'."}])
        print(f"[OK] Vision LLM Check Passed (Text Mode). Response: {response}")
    except Exception as e:
        print(f"[FAIL] Vision LLM Check Failed: {e}")

    print("="*50)
    print("Verification Completed")
    print("="*50)

if __name__ == "__main__":
    verify_connectivity()
