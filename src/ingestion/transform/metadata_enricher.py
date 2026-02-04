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
        prompt = self._prompt_template.format(text=(text or "").strip())
        response = self._llm.chat([{"role": "user", "content": prompt}])
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
        for line in text.splitlines():
            candidate = line.strip()
            if not candidate:
                continue
            candidate = re.sub(r"^\s{0,3}#{1,6}\s+", "", candidate)
            candidate = re.sub(r"^\s*[-*]\s+", "", candidate)
            return self._truncate(candidate, self._cfg.max_title_chars)
        return ""

    def _extract_summary(self, text: str) -> str:
        compact = re.sub(r"\s+", " ", text).strip()
        return self._truncate(compact, self._cfg.max_summary_chars)

    def _extract_tags(self, text: str) -> List[str]:
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
        if max_chars <= 0:
            return ""
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip()
