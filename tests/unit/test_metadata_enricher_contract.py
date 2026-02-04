from unittest.mock import MagicMock

import pytest

from src.core.settings import (
    IngestionSettings,
    MetadataEnricherSettings,
    Settings,
    TransformSettings,
)
from src.ingestion.models import Chunk
from src.ingestion.transform.metadata_enricher import MetadataEnricher
from src.libs.llm.base_llm import BaseLLM


@pytest.fixture
def mock_settings() -> Settings:
    settings = MagicMock(spec=Settings)
    settings.ingestion = MagicMock(spec=IngestionSettings)
    settings.ingestion.transform = MagicMock(spec=TransformSettings)
    settings.ingestion.transform.metadata_enricher = MetadataEnricherSettings(
        enabled=True,
        enable_llm=False,
        fallback_on_error=True,
        max_title_chars=80,
        max_summary_chars=220,
        max_tags=6,
    )
    return settings


def test_rule_mode_outputs_required_keys(mock_settings: Settings) -> None:
    enricher = MetadataEnricher(mock_settings)
    chunk = Chunk(
        text="# 标题\n\n这里是正文内容，包含一些关键术语：RAG pipeline embedding.\n",
        metadata={},
        doc_id="d1",
    )

    enriched = enricher.transform([chunk])[0].metadata

    assert isinstance(enriched.get("title"), str) and enriched["title"].strip()
    assert isinstance(enriched.get("summary"), str) and enriched["summary"].strip()
    assert isinstance(enriched.get("tags"), list) and len(enriched["tags"]) > 0


def test_llm_mode_overrides_metadata(mock_settings: Settings) -> None:
    mock_settings.ingestion.transform.metadata_enricher.enable_llm = True

    llm = MagicMock(spec=BaseLLM)
    llm.chat.return_value = '{"title":"T","summary":"S","tags":["a","b"]}'

    enricher = MetadataEnricher(mock_settings, llm=llm)
    chunk = Chunk(text="Some content.", metadata={}, doc_id="d1")

    enriched = enricher.transform([chunk])[0].metadata

    assert enriched["title"] == "T"
    assert enriched["summary"] == "S"
    assert enriched["tags"] == ["a", "b"]
    assert enriched.get("metadata_enriched_by_llm") is True
    llm.chat.assert_called_once()


def test_llm_failure_falls_back_to_rule(mock_settings: Settings) -> None:
    mock_settings.ingestion.transform.metadata_enricher.enable_llm = True
    mock_settings.ingestion.transform.metadata_enricher.fallback_on_error = True

    llm = MagicMock(spec=BaseLLM)
    llm.chat.side_effect = RuntimeError("LLM down")

    enricher = MetadataEnricher(mock_settings, llm=llm)
    chunk = Chunk(text="Fallback content.", metadata={}, doc_id="d1")

    enriched = enricher.transform([chunk])[0].metadata

    assert isinstance(enriched.get("title"), str) and enriched["title"].strip()
    assert isinstance(enriched.get("summary"), str) and enriched["summary"].strip()
    assert isinstance(enriched.get("tags"), list) and len(enriched["tags"]) > 0
    assert enriched.get("metadata_enriched_by_llm") is False
    assert "metadata_enrichment_error" in enriched
