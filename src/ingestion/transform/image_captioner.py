import logging
from typing import Any, Dict, List, Optional

from src.core.settings import ImageCaptionerSettings, Settings
from src.ingestion.models import Chunk
from src.ingestion.transform.base_transform import BaseTransform, TraceContext
from src.libs.llm.base_llm import BaseLLM
from src.libs.llm.llm_factory import LLMFactory

logger = logging.getLogger(__name__)


class ImageCaptioner(BaseTransform):
    """
    Transform component that generates captions for images referenced in chunks
    using a Vision LLM.
    """

    def __init__(
        self,
        settings: Settings,
        llm: Optional[BaseLLM] = None,
    ):
        self._settings = settings
        self._cfg: ImageCaptionerSettings = settings.ingestion.transform.image_captioner

        # Load prompt
        self._prompt = "Describe this image in detail."
        if self._cfg.prompt_path:
            try:
                # Handle relative paths from project root
                import os

                prompt_path = self._cfg.prompt_path
                if not os.path.isabs(prompt_path):
                    # Assuming CWD is project root, which is typical
                    pass

                if os.path.exists(prompt_path):
                    with open(prompt_path, "r", encoding="utf-8") as f:
                        self._prompt = f.read().strip()
                else:
                    logger.warning(f"Prompt file not found: {prompt_path}")
            except Exception as e:
                logger.warning(
                    f"Failed to load prompt from {self._cfg.prompt_path}: {e}"
                )

        self._llm: Optional[BaseLLM] = llm
        if self._llm is None and self._cfg.enabled:
            try:
                self._llm = LLMFactory.create_vision(settings)
            except Exception as e:
                logger.warning(
                    f"Failed to initialize Vision LLM: {e}. Image captioning disabled."
                )
                self._llm = None

    def transform(
        self, chunks: List[Chunk], trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        """
        Process chunks to add image captions.
        """
        if not chunks:
            return []

        # If disabled or LLM failed to init, return chunks as is
        # (or maybe check per chunk if we want to log 'skipped')
        if not self._cfg.enabled or not self._llm:
            return chunks

        for chunk in chunks:
            self._process_chunk(chunk)
        return chunks

    def _process_chunk(self, chunk: Chunk) -> None:
        """
        Generate captions for images in a single chunk.
        """
        image_refs = chunk.metadata.get("image_refs", [])
        if not image_refs:
            return

        captions: Dict[str, str] = {}
        errors: List[str] = []

        for img_id in image_refs:
            try:
                caption = self._generate_caption(img_id)
                if caption:
                    captions[img_id] = caption
            except Exception as e:
                logger.error(f"Failed to caption image {img_id}: {e}")
                errors.append(f"{img_id}: {str(e)}")
                # If fallback is NOT enabled, we should re-raise
                if not self._cfg.fallback_on_error:
                    raise

        if captions:
            chunk.metadata["image_captions"] = captions

        if errors:
            chunk.metadata["has_unprocessed_images"] = True
            existing_errors = chunk.metadata.get("processing_errors", [])
            if isinstance(existing_errors, list):
                existing_errors.extend(errors)
                chunk.metadata["processing_errors"] = existing_errors
            else:
                chunk.metadata["processing_errors"] = errors

    def _generate_caption(self, img_id: str) -> str:
        """
        Call Vision LLM to generate caption.
        """
        if not self._llm:
            return ""

        # Construct message for Vision LLM.
        # We assume the LLM provider supports 'content' as list for vision tasks.
        # Using a dummy URL for now since we don't have a real image store service yet.
        # In a real implementation, we would read the file and encode base64
        # or provide a valid accessible URL.

        # NOTE: This relies on the specific LLM implementation (e.g. OpenAI)
        # handling the list content correctly despite BaseLLM type hints.
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self._prompt},
                    # Placeholder image URL - likely won't work with real API
                    # without valid URL/base64. But fine for mocking/testing.
                    {"type": "image_url", "image_url": {"url": f"file://{img_id}"}},
                ],
            }
        ]

        return self._llm.chat(messages)  # type: ignore
