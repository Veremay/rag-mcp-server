"""
图注生成：对 chunk 内引用的图片用 Vision LLM 生成 caption 并写回 metadata 与正文。

先按 ![Image](path) 过滤出本块实际出现的图片，再按语言(中/英)调整 prompt 生成描述，
并追加到 chunk.text 供检索匹配；失败时可根据 fallback_on_error 决定是否抛错。
"""
import base64
import logging
import mimetypes
import re
from pathlib import Path
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

    支持从 prompt_path 加载提示词；未启用或 Vision LLM 初始化失败时仅做 metadata.images
    过滤与校正，不生成 caption，保证流水线仍可继续。
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
        Process chunks to add image captions and correct image metadata.

        先按正文中的 ![Image](path) 过滤出本块真实出现的图片并更新 metadata.images，
        再在启用且 LLM 可用时为每张图生成 caption 并写入 metadata 与 chunk.text。
        """
        if not chunks:
            return []

        # Check if caption generation is enabled
        captions_enabled = self._cfg.enabled and self._llm is not None
        
        for chunk in chunks:

            self._process_chunk(chunk, captions_enabled=captions_enabled)
            
        return chunks

    def _process_chunk(self, chunk: Chunk, captions_enabled: bool) -> None:
        """
        Generate captions for images in a single chunk and update metadata.

        用 path/stem 匹配 metadata.images 与正文引用，只保留本块出现的图片；再按语言检测
        生成 caption 并回写到 image 对象与 chunk.text，便于检索时匹配图注语义。
        """
        # Extract image paths from text using regex
        # Matches ![Image](path)
        image_refs = re.findall(r"!\[Image\]\((.*?)\)", chunk.text)
        
        # 1. Update metadata["images"] to only include images actually present in this chunk
        # (Splitter copies document metadata to all chunks, so we need to filter)
        all_images = chunk.metadata.get("images", [])
        
        if not isinstance(all_images, list):
            all_images = []
            
        chunk_images = []
        if image_refs:
            # Normalize refs for comparison
            refs_set = set()
            refs_ids = set()
            for r in image_refs:
                try:
                    p = Path(r).resolve()
                    refs_set.add(str(p))
                    refs_ids.add(p.stem) # Add image ID (filename without ext)
                except Exception:
                    refs_set.add(r)
            
            for img in all_images:
                if not isinstance(img, dict):
                    continue
                img_path = img.get("path")
                img_id = img.get("image_id") # Explicit ID from metadata
                
                match = False
                if img_path:
                    try:
                        normalized_path = str(Path(img_path).resolve())
                        if normalized_path in refs_set:
                            match = True
                        elif Path(img_path).stem in refs_ids:
                             match = True
                    except Exception:
                        if img_path in refs_set:
                             match = True
                
                # Also check explicit image_id if available
                if not match and img_id and img_id in refs_ids:
                    match = True
                    
                if match:
                    chunk_images.append(img)
        
        chunk.metadata["images"] = chunk_images
        
        # 2. Generate captions if enabled and needed
        if not captions_enabled or not image_refs:
            return
    
        captions: Dict[str, str] = {}
        errors: List[str] = []

        # Detect language from chunk text
        language = self._detect_language(chunk.text)

        for img_path in image_refs:
            if not Path(img_path).exists():
                logger.warning(
                    "Image ref in chunk is not an existing file, skip captioning: %s",
                    img_path,
                )
                errors.append("%s: file not found" % (img_path,))
                continue
            try:
                caption = self._generate_caption(img_path, language=language)
                if caption:
                    captions[img_path] = caption
                    # Also update the image object in metadata with caption
                    for img in chunk_images:
                        if str(Path(img.get("path", "")).resolve()) == str(Path(img_path).resolve()):
                            img["caption"] = caption
            except Exception as e:
                logger.error(f"Failed to caption image {img_path}: {e}")
                errors.append(f"{img_path}: {str(e)}")
                # If fallback is NOT enabled, we should re-raise
                if not self._cfg.fallback_on_error:
                    raise

        if captions:
            chunk.metadata["image_captions"] = captions
            # Append captions to chunk text for retrieval context
            caption_text = "\n\n".join([f"Image Caption ({Path(k).name}): {v}" for k, v in captions.items()])
            chunk.text += f"\n\n[Image Captions]\n{caption_text}"

        if errors:
            chunk.metadata["has_unprocessed_images"] = True
            existing_errors = chunk.metadata.get("processing_errors", [])
            if isinstance(existing_errors, list):
                existing_errors.extend(errors)
                chunk.metadata["processing_errors"] = existing_errors
            else:
                chunk.metadata["processing_errors"] = errors

    def _detect_language(self, text: str) -> str:
        """
        Heuristic to detect if text is Chinese or English.
        Defaults to 'zh' (Chinese) to bias towards Chinese output, unless strong English signal is found.

        先去掉图片引用再检测，避免路径/扩展名干扰；无中文且英文字母数超过阈值才判为 en。
        """
        if not text:
            return "zh"
        
        # Remove image references to avoid biasing detection with file paths/extensions
        # Matches ![Image](path)
        clean_text = re.sub(r"!\[Image\]\(.*?\)", "", text)
        
        if not clean_text.strip():
            return "zh"
        
        # Check for Chinese characters range \u4e00-\u9fff
        for char in clean_text:
            if '\u4e00' <= char <= '\u9fff':
                return "zh"
        
        # If no Chinese, check for significant English content
        # Only return 'en' if there are enough English letters, otherwise default to 'zh'
        english_chars = sum(1 for c in clean_text if 'a' <= c.lower() <= 'z')
        if english_chars > 10:  # Threshold to avoid short codes/symbols triggering English
            return "en"
            
        return "zh"

    def _generate_caption(self, img_path: str, language: str = "en") -> str:
        """
        Call Vision LLM to generate caption.
        读入图片并 base64 为 data URL，按 language 追加中/英提示后发送给 Vision LLM。
        """
        if not self._llm:
            return ""

        try:
            with open(img_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            
            mime_type, _ = mimetypes.guess_type(img_path)
            if not mime_type:
                mime_type = "image/jpeg"
                
            data_url = f"data:{mime_type};base64,{encoded_string}"
            
            # Adjust prompt based on language
            prompt = self._prompt
            if language == "zh":
                prompt += "\n请用中文简洁地概括这张图片的内容，不要冗长。"
            else:
                prompt += "\nPlease summarize the image content concisely in English."

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ]
            return self._llm.chat(messages)  # type: ignore
        except Exception as e:
            logger.error(f"Error preparing image for captioning {img_path}: {e}")
            raise
