"""
Embedding 包：分块的稠密与稀疏编码及批量处理。

DenseEncoder 调用 BaseEmbedding 产出向量；SparseEncoder 用正则分词做词频；
BatchProcessor 按 batch_size 分批调用两者并合并结果，便于控制 API 限流与内存。
"""
from src.ingestion.embedding.batch_processor import (
    BatchMetrics,
    BatchProcessor,
    BatchProcessResult,
)
from src.ingestion.embedding.dense_encoder import DenseEncoder
from src.ingestion.embedding.sparse_encoder import SparseEncoder

__all__ = [
    "DenseEncoder",
    "SparseEncoder",
    "BatchProcessor",
    "BatchMetrics",
    "BatchProcessResult",
]
