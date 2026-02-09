from unittest.mock import MagicMock

import pytest

from src.core.settings import ImageCaptionerSettings, Settings
from src.ingestion.models import Chunk
from src.ingestion.transform.image_captioner import ImageCaptioner
from src.libs.llm.base_llm import BaseLLM


class MockLLM(BaseLLM):
    def chat(self, messages, **kwargs):
        return "A description of the image."


@pytest.fixture
def mock_settings():
    settings = MagicMock(spec=Settings)
    # Setup nested structure mocks
    settings.ingestion = MagicMock()
    settings.ingestion.transform = MagicMock()
    settings.ingestion.transform.image_captioner = ImageCaptionerSettings(
        enabled=True,
        fallback_on_error=True,
        prompt_path="config/prompts/image_captioning.txt",
    )

    settings.vision_llm = MagicMock()
    settings.vision_llm.provider = "openai"

    settings.llm = MagicMock()
    settings.llm.api_key = "dummy"

    return settings


def test_image_captioning_success(mock_settings):
    """Test successful caption generation when enabled."""
    mock_llm = MockLLM()
    chunk = Chunk(text="Test chunk", metadata={"image_refs": ["img1"]})

    captioner = ImageCaptioner(mock_settings, llm=mock_llm)
    results = captioner.transform([chunk])

    assert len(results) == 1
    assert "image_captions" in results[0].metadata
    assert (
        results[0].metadata["image_captions"]["img1"] == "A description of the image."
    )


def test_image_captioning_fallback(mock_settings):
    """Test fallback behavior when LLM fails."""
    mock_llm = MagicMock(spec=BaseLLM)
    mock_llm.chat.side_effect = RuntimeError("API Error")

    chunk = Chunk(text="Test chunk", metadata={"image_refs": ["img1"]})

    captioner = ImageCaptioner(mock_settings, llm=mock_llm)
    results = captioner.transform([chunk])

    assert len(results) == 1
    assert "image_captions" not in results[0].metadata
    assert results[0].metadata.get("has_unprocessed_images") is True
    errors = results[0].metadata.get("processing_errors")
    assert errors is not None
    assert any("API Error" in str(e) for e in errors)


def test_image_captioning_disabled(mock_settings):
    """Test that nothing happens when disabled."""
    mock_settings.ingestion.transform.image_captioner.enabled = False

    # Even with a valid LLM, it should skip
    mock_llm = MockLLM()
    chunk = Chunk(text="Test chunk", metadata={"image_refs": ["img1"]})

    captioner = ImageCaptioner(mock_settings, llm=mock_llm)
    results = captioner.transform([chunk])

    assert "image_captions" not in results[0].metadata


def test_no_image_refs(mock_settings):
    """Test that chunks without images are ignored."""
    mock_llm = MockLLM()
    chunk = Chunk(text="Test chunk", metadata={})  # No image_refs

    captioner = ImageCaptioner(mock_settings, llm=mock_llm)
    results = captioner.transform([chunk])

    assert "image_captions" not in results[0].metadata
    assert "has_unprocessed_images" not in results[0].metadata
