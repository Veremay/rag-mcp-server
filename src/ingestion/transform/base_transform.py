"""
Transform 基类：定义分块变换的统一接口。

所有 transform（ChunkRefiner、MetadataEnricher、ImageCaptioner）实现此接口，
便于 Pipeline 用同一方式顺序调用并可替换/扩展新变换。
"""
from abc import ABC, abstractmethod
from typing import Any, List, Optional

from src.ingestion.models import Chunk

TraceContext = Any


class BaseTransform(ABC):
    """抽象基类：接收并返回 List[Chunk]，可选 trace 用于可观测性。"""

    @abstractmethod
    def transform(
        self, chunks: List[Chunk], trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        raise NotImplementedError
