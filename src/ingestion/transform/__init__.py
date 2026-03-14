"""
Transform 包：分块后的处理链（精炼、元数据增强、图注）。

各 transform 接收 List[Chunk] 返回 List[Chunk]，便于 Pipeline 顺序组合；
支持 trace 与可选 LLM，失败时可配置 fallback 不中断整批。
"""
from src.ingestion.transform.base_transform import BaseTransform
from src.ingestion.transform.chunk_refiner import ChunkRefiner
from src.ingestion.transform.metadata_enricher import MetadataEnricher

__all__ = ["BaseTransform", "ChunkRefiner", "MetadataEnricher"]
