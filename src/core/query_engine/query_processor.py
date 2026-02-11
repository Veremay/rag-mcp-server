from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass(frozen=True)
class QueryProcessResult:
    keywords: List[str]
    filters: Dict[str, Any]


class QueryProcessor:
    """
    Query pre-processing for retrieval.

    This component extracts:
    - keywords: token list used by sparse retrieval
    - filters: normalized metadata constraints (may be empty)
    """

    def __init__(
        self,
        *,
        stopwords: Optional[Set[str]] = None,
        allowed_filter_keys: Optional[Set[str]] = None,
    ):
        self._stopwords = stopwords or {
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
        if not query:
            return []

        words = re.findall(r"[A-Za-z]{2,}", query.lower())
        numbers = re.findall(r"\d{2,}", query)
        zh = re.findall(r"[\u4e00-\u9fff]{2,}", query)
        acronyms = re.findall(r"[A-Za-z]{2,}\d+", query.lower())

        tokens: List[str] = []
        for w in words:
            if w in self._stopwords:
                continue
            tokens.append(w)
        tokens.extend(acronyms)
        tokens.extend(numbers)
        tokens.extend(zh)

        seen: Set[str] = set()
        deduped: List[str] = []
        for t in tokens:
            if t in seen:
                continue
            seen.add(t)
            deduped.append(t)
        return deduped
