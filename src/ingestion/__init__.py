"""
Ingestion 包：文档加载、切分、变换、编码与写入向量库/BM25/图片存储的流水线。

对外暴露 Document、Chunk 等模型及流水线组件，供 ingest 脚本与 MCP 等调用方使用，
实现「文件 -> 分块 -> 元数据/图片处理 -> 向量+稀疏 -> 落库」的完整摄取流程。
"""
from .models import Chunk, Document

__all__ = ["Document", "Chunk"]
