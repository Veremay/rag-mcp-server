from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class VectorRecord:
    """Standardized vector record for storage."""

    id: str
    embedding: List[float]
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseVectorStore(ABC):
    """Abstract base class for vector store implementations."""

    @abstractmethod
    def upsert(self, records: List[VectorRecord], trace: Optional[Any] = None) -> None:
        """
        Insert or update records in the vector store.

        Args:
            records: List of VectorRecord objects to store.
            trace: Optional trace context (placeholder for Observability phase).
        """
        pass

    @abstractmethod
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
        pass
