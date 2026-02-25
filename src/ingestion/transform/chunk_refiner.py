import logging
import re
from pathlib import Path
from typing import List, Optional

from src.core.settings import Settings
from src.ingestion.models import Chunk
from src.ingestion.transform.base_transform import BaseTransform, TraceContext
from src.libs.llm.base_llm import BaseLLM
from src.libs.llm.llm_factory import LLMFactory

logger = logging.getLogger(__name__)


class ChunkRefiner(BaseTransform):
    def __init__(self, settings: Settings):
        self.settings = settings
        self.refiner_settings = settings.ingestion.transform.chunk_refiner

        self.llm: Optional[BaseLLM] = None
        if self.refiner_settings.enable_llm:
            try:
                self.llm = LLMFactory.create(settings)
            except Exception as e:
                logger.warning(
                    f"Failed to initialize LLM for ChunkRefiner: {e}. LLM refinement will be disabled."
                )
                self.llm = None

        self.prompt_template = self._load_prompt()

    def _load_prompt(self) -> str:
        path = self.refiner_settings.prompt_path
        try:
            return Path(path).read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(
                f"Failed to load chunk refinement prompt from {path}: {e}. Using default."
            )
            return "Refine the following text for clarity and fix formatting issues:\n\n{text}"

    def transform(
        self, chunks: List[Chunk], trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        refined_chunks = []
        for chunk in chunks:
            self.refine_chunk(chunk)
            refined_chunks.append(chunk)

        return refined_chunks

    def refine_chunk(self, chunk: Chunk) -> Chunk:
        original_text = chunk.text
        text = self._apply_rules(original_text)

        if self.refiner_settings.enable_llm and self.llm:
            try:
                refined_text = self._apply_llm(text)
                
                # Failsafe: Ensure image references are preserved
                # LLM might strip them out as "formatting issues"
                original_refs = re.findall(r"!\[Image\]\([^)]+\)", original_text)
                if original_refs:
                    refined_refs = set(re.findall(r"!\[Image\]\([^)]+\)", refined_text))
                    missing_refs = [ref for ref in original_refs if ref not in refined_refs]
                    
                    if missing_refs:
                        logger.warning(
                            f"Restored {len(missing_refs)} missing image refs for chunk {chunk.id} after refinement"
                        )
                        refined_text += "\n\n" + "\n".join(missing_refs)

                text = refined_text
                chunk.metadata["refined_by_llm"] = True
            except Exception as e:
                logger.error(f"LLM refinement failed for chunk {chunk.id}: {e}")
                if not self.refiner_settings.fallback_on_error:
                    raise e
                chunk.metadata["refinement_error"] = str(e)
                chunk.metadata["refined_by_llm"] = False

        chunk.text = text
        return chunk

    def _apply_rules(self, text: str) -> str:
        if not text:
            return ""

        text = text.strip()

        text = re.sub(
            r"^\s*Page \d+( of \d+)?\s*$", "", text, flags=re.MULTILINE | re.IGNORECASE
        )
        text = re.sub(r"^\s*-\s*\d+\s*-\s*$", "", text, flags=re.MULTILINE)

        return text.strip()

    def _apply_llm(self, text: str) -> str:
        prompt = self.prompt_template.format(text=text)
        messages = [{"role": "user", "content": prompt}]
        llm = self.llm
        if llm is None:
            raise RuntimeError("LLM is not initialized")
        response = llm.chat(messages)
        return response.strip()
