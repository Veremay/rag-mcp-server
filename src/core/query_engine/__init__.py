"""
查询引擎包：对外暴露检索、融合、重排等组件的统一入口。

集中导出 DenseRetriever、SparseRetriever、QueryProcessor、MetadataFilter 等，
便于上层（如 MCP 工具、RAG 管道）只依赖本包即可组装检索流程，避免散落 import。
"""
from src.core.query_engine.dense_retriever import DenseHit, DenseRetriever
from src.core.query_engine.metadata_filter import MetadataFilter
from src.core.query_engine.query_processor import QueryProcessor, QueryProcessResult
from src.core.query_engine.sparse_retriever import SparseHit, SparseRetriever

__all__ = [
    "DenseRetriever",
    "DenseHit",
    "MetadataFilter",
    "QueryProcessor",
    "QueryProcessResult",
    "SparseRetriever",
    "SparseHit",
]
