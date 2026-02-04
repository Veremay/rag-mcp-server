from abc import ABC, abstractmethod
from typing import List, Optional, Any
from src.ingestion.models import Chunk

TraceContext = Any

class BaseTransform(ABC):
    @abstractmethod
    def transform(self, chunks: List[Chunk], trace: Optional[TraceContext] = None) -> List[Chunk]:
        raise NotImplementedError
