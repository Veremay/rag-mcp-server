"""
响应构建器：将检索命中与引用、多模态内容组装成统一 JSON 响应。

先由 CitationGenerator 生成引用列表，再拼 Markdown 正文并追加 MultimodalAssembler
产出的图片块，保证 content 与 structuredContent 一致、前端既可渲染富文本又可取结构化引用。
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from src.core.query_engine.hybrid_search import HybridSearchHit
from src.core.response.citation_generator import CitationGenerator
from src.core.response.multimodal_assembler import MultimodalAssembler

JsonDict = Dict[str, Any]


class ResponseBuilder:
    """
    将 hits 转为 content + structuredContent 的入口。

    支持注入 citation_generator / multimodal_assembler 便于测试或替换实现；
    无命中时返回友好提示并带空引用，避免前端拿到空 content 时报错。
    """

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
        """
        根据 hits 生成完整响应：正文 Markdown、引用列表、可选图片块。

        先生成 citations 再 _build_markdown 是为了让每条命中与引用按索引一一对应；
        content 首项为文本、再 extend 图片，满足部分客户端「先文后图」的展示顺序。
        collection 传给 multimodal 用于按集合解析图片路径。
        """
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
        """
        将 hits 与 citations 拼成带编号、来源、section 的 Markdown 片段列表。

        每条命中用 [idx] source · section 作为标题、_compact_text 截断正文，便于用户
        快速扫读且与 structuredContent.citations 的 id 对齐，方便点击跳转或高亮。
        """
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
    """
    将正文压成单行并截断到 max_chars，末尾加省略号。
    避免响应里塞入过长原文导致卡顿或超限，同时保留足够上下文供用户判断相关性。
    """
    s = str(text or "")
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return ""
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 1].rstrip() + "…"
