"""
查询预处理：从原始 query 中抽取关键词与元数据过滤条件。

先解析 filters（如 collection:xxx）可让检索只命中指定集合，避免全库扫；
再对剩余文本做分词得到 keywords，供稀疏检索使用。中英文混合时对中文做
按停用词切分并保留单字，是为了在 BM25 下兼顾召回与噪声控制。
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass(frozen=True)
class QueryProcessResult:
    """预处理结果：keywords 供稀疏检索，filters 供向量库/过滤层，便于下游统一使用。"""
    keywords: List[str]
    filters: Dict[str, Any]


class QueryProcessor:
    """
    Query pre-processing for retrieval.

    This component extracts:
    - keywords: token list used by sparse retrieval
    - filters: normalized metadata constraints (may be empty)

    查询预处理器：从自然语言 query 中抽取「关键词」和「过滤条件」。
    抽取 filters 是为了让检索限定在指定 collection/doc_type 等，减少无关文档；
    抽取 keywords 是为了稀疏检索能用更干净的词列表，减少停用词带来的噪声。
    支持中英文停用词与可配置的 filter 键，便于按业务扩展。
    """

    def __init__(
        self,
        *,
        stopwords: Optional[Set[str]] = None,
        allowed_filter_keys: Optional[Set[str]] = None,
    ):
        self._stopwords = stopwords or {
            # English Stopwords
            "the",
            "and",
            "for",
            "with",
            "that",
            "this",
            "from",
            "are",
            "was",
            "were",
            "will",
            "shall",
            "can",
            "could",
            "should",
            "would",
            "not",
            "what",
            "why",
            "how",
            "which",
            "who",
            "when",
            "where",
            "a",
            "an",
            "to",
            "of",
            "in",
            "on",
            "at",
            "as",
            "is",
            "it",
            # Chinese Stopwords (Particles & Common Verbs)
            "的",
            "了",
            "和",
            "是",
            "就",
            "都",
            "而",
            "及",
            "与",
            "着",
            "之",
            "用",
            "于",
            "把",
            "在",
            "有",
            # Common Question Words & Suffixes
            "什么",
            "怎么",
            "哪里",
            "为什么",
            "地址",
        }
        self._allowed_filter_keys = allowed_filter_keys or {
            "collection",
            "doc_type",
            "language",
            "time_range",
            "access_level",
        }

    def process(self, query: str, trace: Optional[Any] = None) -> QueryProcessResult:
        """
        Process the input query into keywords and filters.

        Args:
            query: Raw input query string.
            trace: Optional trace context for observability.

        Returns:
            QueryProcessResult containing keywords and filters.

        将原始 query 解析为 keywords 与 filters。先 _extract_filters 再对剩余文本_extract_keywords，
        这样过滤条件不会进入分词，避免把「collection:docs」当成检索词。
        若过滤后无关键词则退回整句分词，保证即使用户只写了过滤条件也有一定召回。
        trace 用于记录阶段耗时与结果便于排查。
        """
        start_ms = time.time() * 1000.0

        normalized = (query or "").strip()
        if not normalized:
            return QueryProcessResult(keywords=[], filters={})

        filters, residual = self._extract_filters(normalized)
        keywords = self._extract_keywords(residual)

        if not keywords:
            keywords = self._extract_keywords(normalized)
        
        end_ms = time.time() * 1000.0
        if trace:
            fn = getattr(trace, "record_stage", None)
            if callable(fn):
                fn(
                    "query_processing",
                    start_ms=start_ms,
                    end_ms=end_ms,
                    data={
                        "original_query": query,
                        "normalized_query": normalized,
                        "extracted_filters": filters,
                        "extracted_keywords": keywords,
                    },
                )

        return QueryProcessResult(keywords=keywords, filters=filters)

    def _extract_filters(self, query: str) -> Tuple[Dict[str, Any], str]:
        """
        从 query 中识别 key:=value 形式的过滤条件并移除，返回过滤字典与剩余文本。
        只允许 _allowed_filter_keys 内的 key，避免任意键注入导致下游解析异常。
        """
        filters: Dict[str, Any] = {}

        def replacer(match: re.Match[str]) -> str:
            key = str(match.group("key")).strip()
            val = str(match.group("val")).strip()
            if key in self._allowed_filter_keys and val:
                filters[key] = val
            return " "

        key_alt = "|".join(sorted(map(re.escape, self._allowed_filter_keys)))
        pattern = re.compile(
            rf"(?P<key>{key_alt})\s*[:=]\s*(?P<val>[^\s]+)", re.IGNORECASE
        )
        residual = pattern.sub(replacer, query)
        residual = re.sub(r"\s+", " ", residual).strip()
        return filters, residual

    def _extract_keywords(self, query: str) -> List[str]:
        """
        从剩余 query 中抽取检索用 token：英文词、数字、缩写、中文连续块及按停用词切分后的子串。
        中文同时保留整块与单字是为了在 BM25 下既匹配长短语又不过度依赖完整词，提高召回。
        """
        if not query:
            return []

        words = re.findall(r"[A-Za-z]{2,}", query.lower())
        numbers = re.findall(r"\d{2,}", query)
        acronyms = re.findall(r"[A-Za-z]{2,}\d+", query.lower())

        # Enhanced Chinese Processing
        # 1. Extract continuous Chinese chunks
        zh_chunks = re.findall(r"[\u4e00-\u9fff]+", query)
        zh_tokens: List[str] = []
        
        # Build stopword regex for Chinese
        # Sort by length desc to match longest stopwords first
        zh_stops = [s for s in self._stopwords if re.search(r"[\u4e00-\u9fff]", s)]
        sorted_stops = sorted(zh_stops, key=len, reverse=True)
        
        if sorted_stops:
            stop_pattern = re.compile("|".join(map(re.escape, sorted_stops)))
        else:
            stop_pattern = None

        for chunk in zh_chunks:
            # Split chunk by Chinese stopwords
            if stop_pattern:
                # filter(None, ...) removes empty strings from split result
                sub_chunks = list(filter(None, stop_pattern.split(chunk)))
            else:
                sub_chunks = [chunk]

            for sub in sub_chunks:
                zh_tokens.append(sub)
                # If sub length > 1, also add unigrams to improve recall
                # (e.g. "刘泽鹏" -> "刘", "泽", "鹏")
                if len(sub) > 1:
                    zh_tokens.extend(list(sub))

        tokens: List[str] = []
        for w in words:
            if w in self._stopwords:
                continue
            tokens.append(w)
        tokens.extend(acronyms)
        tokens.extend(numbers)
        tokens.extend(zh_tokens)

        seen: Set[str] = set()
        deduped: List[str] = []
        for t in tokens:
            if t in seen:
                continue
            seen.add(t)
            deduped.append(t)
        return deduped
