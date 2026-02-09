from unittest.mock import MagicMock, patch

import pytest

from src.core.settings import (
    ChunkRefinerSettings,
    IngestionSettings,
    Settings,
    TransformSettings,
)
from src.ingestion.models import Chunk
from src.ingestion.transform.chunk_refiner import ChunkRefiner


@pytest.fixture
def mock_settings():
    settings = MagicMock(spec=Settings)
    settings.ingestion = MagicMock(spec=IngestionSettings)
    settings.ingestion.transform = MagicMock(spec=TransformSettings)

    refiner_settings = ChunkRefinerSettings(
        enabled=True,
        enable_llm=False,
        llm_provider=None,
        prompt_path="dummy_path.txt",
        fallback_on_error=True,
    )
    settings.ingestion.transform.chunk_refiner = refiner_settings

    settings.llm = MagicMock()
    settings.llm.provider = "openai"
    settings.llm.api_key = "sk-test"

    return settings


def test_rule_based_cleaning(mock_settings):
    with patch("pathlib.Path.read_text", return_value="{text}"):
        refiner = ChunkRefiner(mock_settings)

        text = """
        This is useful content.
        Page 1 of 10
        More content.
        - 5 -
        End.
        """
        chunk = Chunk(text=text)

        refined_chunk = refiner.refine_chunk(chunk)

        assert "Page 1 of 10" not in refined_chunk.text
        assert "- 5 -" not in refined_chunk.text
        assert "This is useful content." in refined_chunk.text
        assert "More content." in refined_chunk.text


def test_llm_refinement(mock_settings):
    mock_settings.ingestion.transform.chunk_refiner.enable_llm = True

    with patch("src.ingestion.transform.chunk_refiner.LLMFactory") as mock_factory:
        mock_llm = MagicMock()
        mock_llm.chat.return_value = "Refined content"
        mock_factory.create.return_value = mock_llm

        with patch("pathlib.Path.read_text", return_value="{text}"):
            refiner = ChunkRefiner(mock_settings)

            chunk = Chunk(text="Raw content")
            refined_chunk = refiner.refine_chunk(chunk)

            assert refined_chunk.text == "Refined content"
            assert refined_chunk.metadata.get("refined_by_llm") is True
            mock_llm.chat.assert_called_once()


def test_llm_fallback(mock_settings):
    mock_settings.ingestion.transform.chunk_refiner.enable_llm = True
    mock_settings.ingestion.transform.chunk_refiner.fallback_on_error = True

    with patch("src.ingestion.transform.chunk_refiner.LLMFactory") as mock_factory:
        mock_llm = MagicMock()
        mock_llm.chat.side_effect = Exception("LLM Error")
        mock_factory.create.return_value = mock_llm

        with patch("pathlib.Path.read_text", return_value="{text}"):
            refiner = ChunkRefiner(mock_settings)

            chunk = Chunk(text="Raw content")
            refined_chunk = refiner.refine_chunk(chunk)

            assert refined_chunk.text == "Raw content"
            assert refined_chunk.metadata.get("refined_by_llm") is False
            assert "refinement_error" in refined_chunk.metadata


def test_transform_list(mock_settings):
    with patch("pathlib.Path.read_text", return_value="{text}"):
        refiner = ChunkRefiner(mock_settings)
        chunks = [Chunk(text=" c1 "), Chunk(text=" c2 ")]

        refined = refiner.transform(chunks)

        assert len(refined) == 2
        assert refined[0].text == "c1"
        assert refined[1].text == "c2"
