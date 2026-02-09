from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from src.core.query_engine.hybrid_search import HybridSearchHit
from src.core.response.citation_generator import CitationGenerator
from src.core.response.multimodal_assembler import MultimodalAssembler

JsonDict = Dict[str, Any]


class ResponseBuilder:
    def __init__(
        self,
        *,
        citation_generator: Optional[CitationGenerator] = None,
        multimodal_assembler: Optional[MultimodalAssembler] = None,
    ) -> None:
        self._citations = citation_generator or CitationGenerator()
        self._multimodal = multimodal_assembler or MultimodalAssembler()

    def build(
        self,
        hits: Sequence[HybridSearchHit],
        *,
        query: str,
        collection: Optional[str] = None,
    ) -> JsonDict:
        normalized_query = (query or "").strip()
        if not hits:
            msg = "未找到相关文档，请先运行 ingest.py 摄取数据"
            return {
                "content": [{"type": "text", "text": msg}],
                "structuredContent": {"answer": msg, "citations": []},
            }

        citations = self._citations.generate(hits)
        markdown = self._build_markdown(
            hits, citations=citations, query=normalized_query
        )
        content: List[JsonDict] = [{"type": "text", "text": markdown}]
        content.extend(self._multimodal.assemble(hits, collection=collection))
        return {
            "content": content,
            "structuredContent": {"answer": markdown, "citations": citations},
        }

    def _build_markdown(
        self,
        hits: Sequence[HybridSearchHit],
        *,
        citations: List[JsonDict],
        query: str,
    ) -> str:
        lines: List[str] = []
        if query:
            lines.append(f"查询：{query}")
            lines.append("")

        for idx, hit in enumerate(hits, start=1):
            snippet = _compact_text(
                getattr(getattr(hit, "record", None), "content", "")
            )
            source = str((citations[idx - 1].get("source") or "")).strip()
            section = str(
                (getattr(getattr(hit, "record", None), "metadata", {}) or {}).get(
                    "section_path", ""
                )
            ).strip()

            header = f"[{idx}]"
            if source:
                header = f"{header} {source}"
            if section:
                header = f"{header} · {section}"

            lines.append(header)
            if snippet:
                lines.append(snippet)
            lines.append("")

        return "\n".join(lines).strip() + "\n"


def _compact_text(text: Any, *, max_chars: int = 380) -> str:
    s = str(text or "")
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return ""
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 1].rstrip() + "…"
