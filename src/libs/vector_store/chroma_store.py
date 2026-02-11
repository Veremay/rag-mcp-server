from __future__ import annotations

from typing import Any, Dict, List, Optional, cast

from src.core.settings import Settings
from src.libs.vector_store.base_vector_store import BaseVectorStore, VectorRecord


class ChromaStore(BaseVectorStore):
    """
    ChromaDB implementation of the VectorStore interface.
    """

    def __init__(self, settings: Settings):
        """
        Initialize the ChromaStore with settings.

        Args:
            settings: Global settings object containing vector_store configuration.
        """
        self.persist_path = settings.vector_store.persist_path
        self.collection_name = settings.vector_store.collection_name

        try:
            import chromadb  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "vector_store.backend=chroma 需要安装 chromadb。"
                "请执行：pip install chromadb"
                "；或将 config/settings.yaml 中 vector_store.backend 改为 jsonl。"
            ) from e

        self.client = chromadb.PersistentClient(path=self.persist_path)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, records: List[VectorRecord], trace: Optional[Any] = None) -> None:
        """
        Insert or update records in the vector store.

        Args:
            records: List of VectorRecord objects to store.
            trace: Optional trace context (placeholder for Observability phase).
        """
        if not records:
            return

        ids = [r.id for r in records]
        embeddings = [r.embedding for r in records]
        documents = [r.content for r in records]
        metadatas = [r.metadata for r in records]

        # Upsert handles both insert and update
        self.collection.upsert(
            ids=ids,
            embeddings=cast(Any, embeddings),
            documents=documents,
            metadatas=cast(Any, metadatas),
        )

    def query(
        self,
        vector: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional[Any] = None,
    ) -> List[VectorRecord]:
        """
        Query the vector store for similar records.

        Args:
            vector: Query embedding vector.
            top_k: Number of results to return.
            filters: Optional metadata filters.
            trace: Optional trace context.

        Returns:
            List of VectorRecord objects ordered by similarity.
        """
        effective_filters = filters or None
        results = cast(
            Any,
            self.collection.query(
                query_embeddings=cast(Any, [vector]),
                n_results=top_k,
                where=effective_filters,
                include=["embeddings", "documents", "metadatas", "distances"],
            ),
        )

        # Chroma returns lists of lists (one list per query embedding)
        # We only queried one embedding, so we take the first element
        if not results["ids"] or len(results["ids"][0]) == 0:
            return []

        ids = results["ids"][0]
        embeddings = results["embeddings"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        # distances = results["distances"][0] # Can be used if we want to return scores

        records: List[VectorRecord] = []
        for i in range(len(ids)):
            raw_metadata = metadatas[i] if metadatas[i] else {}
            metadata: Dict[str, Any] = (
                dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
            )
            records.append(
                VectorRecord(
                    id=ids[i],
                    embedding=list(embeddings[i]),
                    content=documents[i],
                    metadata=metadata,
                )
            )

        return records

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            count = self.collection.count()
        except Exception:
            count = -1
            
        return {
            "count": count,
            "backend": "chroma",
            "collection_name": self.collection_name,
            "persist_path": self.persist_path,
        }

    def delete_by_metadata(self, filters: Dict[str, Any]) -> None:
        """
        Delete records matching the given metadata filters.

        Args:
            filters: Metadata filters to match records to delete.
        """
        if not filters:
            return
        self.collection.delete(where=filters)

    def get_records_by_metadata(self, filters: Dict[str, Any]) -> List[VectorRecord]:
        """
        Get records matching the given metadata filters.

        Args:
            filters: Metadata filters to match records.

        Returns:
            List of VectorRecord objects.
         """
        # If filters is None or empty dict, we assume "no filter" and fetch all records.
        # Note: Fetching all records might be heavy for large collections.
        where_clause = filters if filters else None
        
        results = self.collection.get(where=where_clause, include=["metadatas", "documents", "embeddings"])
        if not results or not results["ids"]:
            return []
            
        records = []
        # Chroma returns lists of items
        count = len(results["ids"])
        for i in range(count):
            records.append(VectorRecord(
                id=results["ids"][i],
                embedding=results["embeddings"][i] if results.get("embeddings") is not None else [],
                content=results["documents"][i] if results.get("documents") is not None else "",
                metadata=results["metadatas"][i] if results.get("metadatas") is not None else {},
            ))
        return records

