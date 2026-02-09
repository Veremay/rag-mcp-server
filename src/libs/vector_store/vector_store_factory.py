import os
import sys

from src.core.settings import Settings
from src.libs.vector_store.base_vector_store import BaseVectorStore


class VectorStoreFactory:
    """Factory for creating vector store instances based on configuration."""

    @staticmethod
    def create(settings: Settings) -> BaseVectorStore:
        """
        Create a vector store instance.

        Args:
            settings: Global settings object containing vector_store configuration.

        Returns:
            An instance of BaseVectorStore.

        Raises:
            ValueError: If the configured backend is not supported.
        """
        backend = settings.vector_store.backend.lower()

        if backend == "chroma":
            if os.name == "nt" and os.getenv(
                "FORCE_CHROMA", ""
            ).strip().lower() not in {
                "1",
                "true",
                "yes",
            }:
                from src.libs.vector_store.jsonl_store import JsonlStore

                print(
                    "WARN: 检测到 Windows 环境下 chromadb 可能发生崩溃，"
                    "已自动回退到 jsonl 向量存储（可设置 FORCE_CHROMA=1 强制使用 chroma）。",
                    file=sys.stderr,
                )
                old_backend = settings.vector_store.backend
                old_persist_path = settings.vector_store.persist_path
                try:
                    settings.vector_store.backend = "jsonl"
                    settings.vector_store.persist_path = "./data/db/jsonl"
                    return JsonlStore(settings)
                finally:
                    settings.vector_store.backend = old_backend
                    settings.vector_store.persist_path = old_persist_path

            from src.libs.vector_store.chroma_store import ChromaStore

            return ChromaStore(settings)

        if backend == "jsonl":
            from src.libs.vector_store.jsonl_store import JsonlStore

            return JsonlStore(settings)

        # Extension point for other backends (e.g., qdrant, pinecone)

        raise ValueError(f"Unsupported vector store backend: {backend}")
