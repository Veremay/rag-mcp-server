"""
元数据增强：为每个 chunk 补充 title、summary、tags。

先规则抽取(标题/摘要/词频 tag)，再可选 LLM 生成更高质量元数据；LLM 失败时
可根据 fallback_on_error 决定是否抛错。增强后的 metadata 供检索与展示使用。
"""
import json
import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from src.core.settings import MetadataEnricherSettings, Settings
from src.ingestion.models import Chunk
from src.ingestion.transform.base_transform import BaseTransform, TraceContext
from src.libs.llm.base_llm import BaseLLM
from src.libs.llm.llm_factory import LLMFactory

logger = logging.getLogger(__name__)


class MetadataEnricher(BaseTransform):
    """
    规则 + 可选 LLM 的元数据增强。enabled 关闭时直接返回原 chunks；
    _rule_enrich 保证至少 title/summary/tags 有默认值，_llm_enrich 再覆盖并做长度与条数限制。
    """

    def __init__(
        self,
        settings: Settings,
        llm: Optional[BaseLLM] = None,
        prompt_template: Optional[str] = None,
    ):
        self._settings = settings
        self._cfg: MetadataEnricherSettings = (
            settings.ingestion.transform.metadata_enricher
        )
        self._prompt_template = prompt_template or (
            "You are a metadata generator for a RAG chunk.\n"
            "Given the chunk text, return STRICT JSON with keys:\n"
            "- title: string\n"
            "- summary: string\n"
            "- tags: array of strings\n"
            "Rules:\n"
            "- title and summary must be non-empty.\n"
            "- tags must be 1-10 short strings.\n"
            "- Return JSON only.\n\n"
            "CHUNK:\n{text}\n"
        )

        self._llm: Optional[BaseLLM] = llm
        if self._llm is None and self._cfg.enable_llm:
            try:
                self._llm = LLMFactory.create(settings)
            except Exception as e:
                logger.warning(
                    f"Failed to initialize LLM for MetadataEnricher: {e}. LLM enrichment disabled."
                )
                self._llm = None

    def transform(
        self, chunks: List[Chunk], trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        """逐块 enrich，未开启或空列表时直接返回，避免多余开销。"""
        if not chunks:
            return []

        if not self._cfg.enabled:
            return chunks

        for chunk in chunks:
            self.enrich(chunk)
        return chunks

    def enrich(self, chunk: Chunk) -> Chunk:
        base_metadata = self._rule_enrich(chunk.text)
        chunk.metadata.update(base_metadata)

        if self._cfg.enable_llm and self._llm is not None:
            try:
                llm_metadata = self._llm_enrich(chunk.text)
                chunk.metadata.update(llm_metadata)
                chunk.metadata["metadata_enriched_by_llm"] = True
            except Exception as e:
                logger.error(
                    f"LLM metadata enrichment failed for chunk {chunk.id}: {e}"
                )
                if not self._cfg.fallback_on_error:
                    raise
                chunk.metadata["metadata_enriched_by_llm"] = False
                chunk.metadata["metadata_enrichment_error"] = str(e)

        return chunk

    def _rule_enrich(self, text: str) -> Dict[str, Any]:
        """规则抽取 title/summary/tags，无结果时给默认值保证下游不报错。"""
        cleaned = (text or "").strip()
        title = self._extract_title(cleaned)
        summary = self._extract_summary(cleaned)
        tags = self._extract_tags(cleaned)

        if not title:
            title = "Untitled"
        if not summary:
            summary = title
        if not tags:
            tags = [title[: min(len(title), 24)]]

        return {"title": title, "summary": summary, "tags": tags}

    def _llm_enrich(self, text: str) -> Dict[str, Any]:
        """调用 LLM 生成 JSON 元数据，校验 schema 并做截断与条数限制。"""
        prompt = self._prompt_template.format(text=(text or "").strip())
        llm = self._llm
        if llm is None:
            raise RuntimeError("LLM is not initialized")
        response = llm.chat([{"role": "user", "content": prompt}])
        payload = self._parse_llm_json(response)

        title = payload.get("title")
        summary = payload.get("summary")
        tags = payload.get("tags")

        if not isinstance(title, str) or not title.strip():
            raise ValueError(
                "LLM metadata schema error: title must be a non-empty string"
            )
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError(
                "LLM metadata schema error: summary must be a non-empty string"
            )
        if (
            not isinstance(tags, list)
            or not tags
            or any((not isinstance(t, str) or not t.strip()) for t in tags)
        ):
            raise ValueError(
                "LLM metadata schema error: tags must be a non-empty string list"
            )

        title = self._truncate(title.strip(), self._cfg.max_title_chars)
        summary = self._truncate(summary.strip(), self._cfg.max_summary_chars)
        tags = [self._truncate(t.strip(), 32) for t in tags][: self._cfg.max_tags]

        return {"title": title, "summary": summary, "tags": tags}

    def _parse_llm_json(self, text: str) -> Dict[str, Any]:
        """从 LLM 回复中提取第一个 JSON 对象，避免前后说明文字导致解析失败。"""
        raw = (text or "").strip()

        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            raise ValueError(
                "LLM metadata schema error: cannot find JSON object in response"
            )

        try:
            return json.loads(match.group(0))
        except Exception as e:
            raise ValueError(f"LLM metadata schema error: invalid JSON ({e})") from e

    def _extract_title(self, text: str) -> str:
        """取首条非空行并去掉 Markdown 标题符与列表符，截断到配置长度。"""
        for line in text.splitlines():
            candidate = line.strip()
            if not candidate:
                continue
            candidate = re.sub(r"^\s{0,3}#{1,6}\s+", "", candidate)
            candidate = re.sub(r"^\s*[-*]\s+", "", candidate)
            return self._truncate(candidate, self._cfg.max_title_chars)
        return ""

    def _extract_summary(self, text: str) -> str:
        """整段压成单行并截断，作为规则摘要。"""
        compact = re.sub(r"\s+", " ", text).strip()
        return self._truncate(compact, self._cfg.max_summary_chars)

    def _extract_tags(self, text: str) -> List[str]:
        """对前 800 字做分词与词频排序，取 top max_tags 作为规则 tag。"""
        if not text:
            return []

        tokens = self._tokenize(text[:800])
        if not tokens:
            return []

        counts = Counter(tokens)
        ranked = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
        tags = [t for t, _ in ranked[: self._cfg.max_tags]]
        return tags

    def _tokenize(self, text: str) -> List[str]:
        """英文 3+ 字母与中文 2+ 字，过滤停用词，供 _extract_tags 词频统计。"""
        stopwords = {
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
        }

        words = re.findall(r"[A-Za-z]{3,}", text.lower())
        zh = re.findall(r"[\u4e00-\u9fff]{2,}", text)

        filtered: List[str] = []
        for w in words:
            if w in stopwords:
                continue
            filtered.append(w)
        filtered.extend(zh)
        return filtered

    def _truncate(self, text: str, max_chars: int) -> str:
        """按字符数截断并 rstrip，避免超长 title/summary/tag 撑破存储或展示。"""
        if max_chars <= 0:
            return ""
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip()
