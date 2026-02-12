#!/usr/bin/env python3
"""
Rebuild BM25 index from existing Vector Store data.
This script is used to:
1. Fetch all records from the Vector Store (ChromaDB)
2. Re-calculate sparse vectors (Sparse Embedding)
3. Incrementally update (Upsert) the BM25 index

Usage:
    python scripts/rebuild_bm25.py [--collection NAME] [--verbose]
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.settings import Settings, load_settings
from src.libs.vector_store.vector_store_factory import VectorStoreFactory
from src.ingestion.embedding.sparse_encoder import SparseEncoder
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.ingestion.models import Chunk
from src.core.trace.trace_context import TraceContext

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("rebuild_bm25")

def main():
    parser = argparse.ArgumentParser(description="Rebuild BM25 index from Vector Store")
    parser.add_argument("--collection", type=str, help="Target collection name (optional, defaults to settings)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    try:
        # 1. Load settings
        settings = load_settings()
        collection_name = args.collection or settings.vector_store.collection_name
        
        logger.info(f"Target Collection: {collection_name}")
        logger.info(f"Vector Store Path: {settings.vector_store.persist_path}")
        # Default BM25 path
        bm25_path = "data/db/bm25"
        logger.info(f"BM25 Index Path: {bm25_path}")

        # 2. Initialize components
        # Vector Store (Source of Truth)
        # Use Factory to respect platform-specific fallbacks (e.g. JSONL on Windows)
        store = VectorStoreFactory.create(settings)
        logger.info(f"Using Vector Store Backend: {type(store).__name__}")
        
        # Sparse Encoder (Compute sparse vectors)
        encoder = SparseEncoder()
        
        # BM25 Indexer (Target index)
        indexer = BM25Indexer(base_dir=bm25_path)

        # Check stats
        try:
            stats = store.get_collection_stats()
            logger.info(f"Collection Stats: {stats}")
        except Exception as e:
            logger.warning(f"Could not get collection stats: {e}")

        # 3. Fetch all records
        logger.info("Fetching records from Vector Store...")
        # Empty filter means fetch all
        try:
            # Some stores might not implement get_records_by_metadata efficiently or at all
            if hasattr(store, "get_records_by_metadata"):
                records = store.get_records_by_metadata(filters={})
            else:
                # Fallback: try to read all if possible, or fail
                logger.error(f"Store {type(store).__name__} does not support get_records_by_metadata")
                return
        except Exception as e:
            logger.error(f"Error fetching records: {e}")
            return
        
        if not records:
            logger.warning(f"No records found in collection '{collection_name}'. Nothing to rebuild.")
            return

        logger.info(f"Found {len(records)} records. Starting processing...")

        # 4. Convert to Chunks
        chunks: List[Chunk] = []
        chunk_ids: List[str] = []
        
        for record in records:
            # Reconstruct Chunk from record
            # Note: doc_id in Chunk might be different from chunk_id (record.id)
            # But here we need to map chunk_id to sparse vector
            chunk = Chunk(
                text=record.content,
                metadata=record.metadata,
                doc_id=record.metadata.get("doc_id", "unknown"),
                # We can store the actual chunk_id in metadata if needed, 
                # but for encoding only text matters.
            )
            chunks.append(chunk)
            chunk_ids.append(record.id)

        # 5. Compute Sparse Vectors
        logger.info("Computing sparse vectors (tokenization & counting)...")
        # We can use a dummy trace or None
        sparse_vectors = encoder.encode(chunks, trace=None)
        
        # 6. Upsert to BM25 Index
        logger.info("Upserting to BM25 Index...")
        indexer.upsert(
            collection=collection_name,
            chunk_ids=chunk_ids,
            sparse_vectors=sparse_vectors
        )
        
        logger.info("✅ BM25 Index Rebuild Complete!")
        logger.info(f"Processed {len(chunks)} chunks.")

    except Exception as e:
        logger.error(f"Failed to rebuild BM25 index: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
