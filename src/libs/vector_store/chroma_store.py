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

        无过滤时部分 Chroma 版本对 get(where=None) 会报 "Error finding id"，
        因此改为先 count() 再 get(limit=count)，用有界查询避免该内部错误。
        """
        where_clause = filters if filters else None
        include = ["metadatas", "documents", "embeddings"]

        if where_clause is not None:
            # 有过滤条件：直接 get(where=...)
            results = self.collection.get(where=where_clause, include=include)
        else:
            # 无过滤（拉取全部）：先取总数再 get(limit=N)，避免无界 get 触发 Chroma 内部错误
            try:
                total = self.collection.count()
            except Exception:
                total = 0
            if total <= 0:
                return []
            results = self.collection.get(limit=total, include=include)

        # 避免对 Chroma 返回的 numpy 数组做布尔判断（如 not results["ids"]），否则报 "truth value of an array is ambiguous"
        if results is None:
            return []
        ids = results.get("ids")
        if ids is None or len(ids) == 0:
            return []

        records = []
        n = len(ids)
        # 用 is not None 代替 or []，避免对 numpy 数组做布尔判断触发 ambiguous 错误
        embeddings = results.get("embeddings")
        documents = results.get("documents")
        metadatas = results.get("metadatas")
        if embeddings is None:
            embeddings = []
        if documents is None:
            documents = []
        if metadatas is None:
            metadatas = []
        for i in range(n):
            emb = list(embeddings[i]) if i < len(embeddings) else []
            doc = documents[i] if i < len(documents) else ""
            meta = metadatas[i] if i < len(metadatas) else {}
            metadata = dict(meta) if isinstance(meta, dict) else {}
            records.append(
                VectorRecord(
                    id=str(ids[i]) if i < len(ids) else "",
                    embedding=emb,
                    content=doc,
                    metadata=metadata,
                )
            )
        return records

