from __future__ import annotations

import streamlit as st

from src.core.settings import load_settings
from src.libs.vector_store.vector_store_factory import VectorStoreFactory
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.ingestion.storage.image_storage import ImageStorage
from src.libs.loader.file_integrity import FileIntegrityRegistry
from src.ingestion.document_manager import DocumentManager
from src.ingestion.pipeline import IngestionPipeline


@st.cache_resource
def get_document_manager() -> DocumentManager:
    settings = load_settings()
    
    # Initialize VectorStore
    vector_store = VectorStoreFactory.create(settings)
    
    # Initialize BM25Indexer
    # Using default path as defined in BM25Indexer.__init__ ("data/db/bm25")
    # This must match what ingestion pipeline uses.
    bm25_indexer = BM25Indexer()
    
    # Initialize ImageStorage
    # Using default path as defined in ImageStorage.__init__ ("data/images")
    image_storage = ImageStorage()
    
    # Initialize FileIntegrityRegistry
    # Using default path as defined in FileIntegrityRegistry.__init__ ("data/cache/ingestion_history.json")
    file_integrity = FileIntegrityRegistry()
    
    return DocumentManager(
        vector_store=vector_store,
        bm25_indexer=bm25_indexer,
        image_storage=image_storage,
        file_integrity=file_integrity
    )


@st.cache_resource
def get_ingestion_pipeline() -> IngestionPipeline:
    """
    Get cached IngestionPipeline instance.
    Note: Pipeline components are stateless or thread-safe enough for Streamlit.
    """
    settings = load_settings()
    return IngestionPipeline(settings)
