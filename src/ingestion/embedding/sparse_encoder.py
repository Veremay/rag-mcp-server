"""
稀疏编码器：对每个 chunk 的 text 按正则分词并统计词频，产出 term -> tf 的 dict 列表。

与 BM25 索引的稀疏向量格式一致，供 BatchProcessor 调用并写入 BM25Indexer；
aencode 直接复用 encode，因本地计算无异步 IO 需求。
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional

from src.ingestion.models import Chunk


class SparseEncoder:
    """
    用 token_pattern 对正文分词，小写化后统计词频，输出 List[Dict[str, float]]。
    与 DenseEncoder 输出条数一致，便于 Pipeline 与 BM25 upsert 一一对应。
    """

    def __init__(self, token_pattern: str = r"\w+"):
        self._token_re = re.compile(token_pattern, flags=re.UNICODE)

    def encode(
        self, chunks: List[Chunk], trace: Optional[Any] = None
    ) -> List[Dict[str, float]]:
        """逐 chunk 分词并词频统计，返回与 chunks 同序的稀疏向量列表。"""
        if not chunks:
            return []

        outputs: List[Dict[str, float]] = []
        for chunk in chunks:
            tokens = [t.lower() for t in self._token_re.findall(chunk.text or "")]
            counts = Counter(tokens)
            outputs.append({term: float(freq) for term, freq in counts.items()})

        if len(outputs) != len(chunks):
            raise ValueError(
                f"SparseEncoder output count mismatch: chunks={len(chunks)} outputs={len(outputs)}"
            )

        return outputs

    async def aencode(
        self, chunks: List[Chunk], trace: Optional[Any] = None
    ) -> List[Dict[str, float]]:
        """同步实现即可，无 IO 等待；保留接口与 DenseEncoder 一致供 BatchProcessor 调用。"""
        return self.encode(chunks, trace=trace)
