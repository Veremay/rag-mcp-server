import os

import pytest
import yaml

from src.core.settings import LLMSettings, Settings, load_settings


@pytest.fixture
def valid_config_path(tmp_path):
    config = {
        "llm": {"provider": "openai", "model": "gpt-4o", "api_key": "test_key"},
        "embedding": {"provider": "openai", "model": "text-embedding-3-small"},
        "vision_llm": {"provider": "azure", "model": "gpt-4o"},
        "vector_store": {"backend": "chroma", "persist_path": "./db"},
        "ingestion": {
            "splitter": {
                "provider": "recursive",
                "chunk_size": 1000,
                "chunk_overlap": 200,
            },
            "transform": {
                "chunk_refiner": {},
                "metadata_enricher": {},
                "image_captioner": {},
            },
        },
        "retrieval": {
            "sparse_backend": "bm25",
            "fusion_algorithm": "rrf",
            "top_k_dense": 10,
            "top_k_sparse": 10,
            "top_k_final": 5,
        },
        "rerank": {"backend": "none", "model": "none", "top_m": 0},
        "evaluation": {"backends": ["custom"], "golden_test_set": "test.json"},
        "observability": {
            "enabled": False,
            "log_file": "test.log",
            "dashboard_port": 8501,
        },
    }
    path = tmp_path / "settings.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f)
    return str(path)


@pytest.fixture
def invalid_config_path(tmp_path):
    # Missing llm provider
    config = {"llm": {"model": "gpt-4o"}, "embedding": {"provider": "openai"}}
    path = tmp_path / "invalid_settings.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f)
    return str(path)


def test_load_valid_settings(valid_config_path):
    settings = load_settings(valid_config_path)
    assert isinstance(settings, Settings)
    assert settings.llm.provider == "openai"
    assert settings.llm.model == "gpt-4o"
    assert settings.vector_store.backend == "chroma"


def test_load_missing_file():
    with pytest.raises(FileNotFoundError):
        load_settings("non_existent_file.yaml")


def test_load_invalid_settings(invalid_config_path):
    with pytest.raises(ValueError) as excinfo:
        load_settings(invalid_config_path)
    # The error message might vary based on implementation order,
    # but it should complain about missing sections or fields
    assert "Missing required" in str(excinfo.value)


def test_load_empty_file(tmp_path):
    path = tmp_path / "empty.yaml"
    with open(path, "w") as f:
        pass

    with pytest.raises(ValueError) as excinfo:
        load_settings(str(path))
    assert "empty" in str(excinfo.value)


def test_load_settings_expands_env_vars(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY1", "test-dashscope-key")

    config = {
        "llm": {
            "provider": "openai",
            "model": "gpt-4o",
            "api_key": "test_key",
        },
        "embedding": {
            "provider": "openai",
            "model": "text-embedding-v4",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "api_key": "${DASHSCOPE_API_KEY1}",
        },
        "vision_llm": {
            "provider": "azure",
            "model": "gpt-4o",
        },
        "vector_store": {
            "backend": "chroma",
            "persist_path": "./db",
        },
        "ingestion": {
            "splitter": {
                "provider": "recursive",
                "chunk_size": 1000,
                "chunk_overlap": 200,
            },
            "transform": {
                "chunk_refiner": {},
                "metadata_enricher": {},
                "image_captioner": {},
            },
        },
        "retrieval": {
            "sparse_backend": "bm25",
            "fusion_algorithm": "rrf",
            "top_k_dense": 10,
            "top_k_sparse": 10,
            "top_k_final": 5,
        },
        "rerank": {
            "backend": "none",
            "model": "none",
            "top_m": 0,
        },
        "evaluation": {
            "backends": ["custom"],
            "golden_test_set": "test.json",
        },
        "observability": {
            "enabled": False,
            "log_file": "test.log",
            "dashboard_port": 8501,
        },
    }

    path = tmp_path / "settings.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f)

    settings = load_settings(str(path))
    assert settings.embedding.api_key == "test-dashscope-key"
