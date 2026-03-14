"""
Storage 包：向量写入、BM25 索引、图片存储等摄取落库组件。

VectorUpserter 将 chunks+向量转为 VectorRecord 写入向量库；BM25Indexer 维护稀疏索引；
ImageStorage 按 collection 存图片并维护 index.json，供检索与 Response 多模态组装使用。
"""
__all__ = []
