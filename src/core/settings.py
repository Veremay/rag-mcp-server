import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _load_dotenv() -> None:
    try:
        from dotenv import find_dotenv, load_dotenv  # type: ignore
    except ImportError:
        return

    path = find_dotenv(usecwd=True)
    if path:
        load_dotenv(path, override=False)


def _expand_env_vars(value: Any) -> Any:
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda m: os.getenv(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(v) for v in value]
    return value


@dataclass
class LLMSettings:
    provider: str
    model: str
    azure_endpoint: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None


@dataclass
class EmbeddingSettings:
    provider: str
    model: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None


@dataclass
class VisionLLMSettings:
    provider: str
    model: str


@dataclass
class VectorStoreSettings:
    backend: str
    persist_path: str
    collection_name: str = "knowledge_hub"


@dataclass
class RetrievalSettings:
    sparse_backend: str
    fusion_algorithm: str
    top_k_dense: int
    top_k_sparse: int
    top_k_final: int


@dataclass
class RerankSettings:
    backend: str
    model: str
    top_m: int


@dataclass
class EvaluationSettings:
    backends: List[str]
    golden_test_set: str


@dataclass
class ObservabilitySettings:
    enabled: bool
    log_file: str
    dashboard_port: int


@dataclass
class SplitterSettings:
    provider: str
    chunk_size: int = 1000
    chunk_overlap: int = 200


@dataclass
class ChunkRefinerSettings:
    enabled: bool = True
    enable_llm: bool = False
    llm_provider: Optional[str] = None  # If None, use default LLM
    prompt_path: str = "config/prompts/chunk_refinement.txt"
    fallback_on_error: bool = True


@dataclass
class MetadataEnricherSettings:
    enabled: bool = True
    enable_llm: bool = False
    llm_provider: Optional[str] = None  # If None, use default LLM
    fallback_on_error: bool = True
    max_title_chars: int = 80
    max_summary_chars: int = 220
    max_tags: int = 6


@dataclass
class ImageCaptionerSettings:
    enabled: bool = False
    prompt_path: str = "config/prompts/image_captioning.txt"
    fallback_on_error: bool = True


@dataclass
class TransformSettings:
    chunk_refiner: ChunkRefinerSettings = field(default_factory=ChunkRefinerSettings)
    metadata_enricher: MetadataEnricherSettings = field(
        default_factory=MetadataEnricherSettings
    )
    image_captioner: ImageCaptionerSettings = field(
        default_factory=ImageCaptionerSettings
    )


@dataclass
class IngestionSettings:
    splitter: SplitterSettings
    transform: TransformSettings


@dataclass
class Settings:
    llm: LLMSettings
    embedding: EmbeddingSettings
    vision_llm: VisionLLMSettings
    vector_store: VectorStoreSettings
    ingestion: IngestionSettings
    retrieval: RetrievalSettings

    rerank: RerankSettings
    evaluation: EvaluationSettings
    observability: ObservabilitySettings


def load_settings(config_path: str = "config/settings.yaml") -> Settings:
    """Load settings from a YAML file."""
    _load_dotenv()
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    if not config_data:
        raise ValueError("Configuration file is empty")

    config_data = _expand_env_vars(config_data)
    validate_settings(config_data)

    ingestion_data = config_data.get("ingestion", {})
    splitter_data = ingestion_data.get("splitter", {})
    transform_data = ingestion_data.get("transform", {})
    chunk_refiner_data = transform_data.get("chunk_refiner", {})
    metadata_enricher_data = transform_data.get("metadata_enricher", {})
    image_captioner_data = transform_data.get("image_captioner", {})

    return Settings(
        llm=LLMSettings(**config_data.get("llm", {})),
        embedding=EmbeddingSettings(**config_data.get("embedding", {})),
        vision_llm=VisionLLMSettings(**config_data.get("vision_llm", {})),
        vector_store=VectorStoreSettings(**config_data.get("vector_store", {})),
        ingestion=IngestionSettings(
            splitter=SplitterSettings(**splitter_data),
            transform=TransformSettings(
                chunk_refiner=ChunkRefinerSettings(**chunk_refiner_data),
                metadata_enricher=MetadataEnricherSettings(**metadata_enricher_data),
                image_captioner=ImageCaptionerSettings(**image_captioner_data),
            ),
        ),
        retrieval=RetrievalSettings(**config_data.get("retrieval", {})),
        rerank=RerankSettings(**config_data.get("rerank", {})),
        evaluation=EvaluationSettings(**config_data.get("evaluation", {})),
        observability=ObservabilitySettings(**config_data.get("observability", {})),
    )


def validate_settings(config_data: Dict[str, Any]) -> None:
    """Validate critical configuration fields."""
    required_sections = [
        "llm",
        "embedding",
        "vision_llm",
        "vector_store",
        "ingestion",
        "retrieval",
        "rerank",
        "evaluation",
        "observability",
    ]

    for section in required_sections:
        if section not in config_data:
            raise ValueError(f"Missing required configuration section: {section}")

    # Validate LLM
    if "provider" not in config_data["llm"]:
        raise ValueError("Missing required field: llm.provider")
    if "model" not in config_data["llm"]:
        raise ValueError("Missing required field: llm.model")

    # Validate Embedding
    if "provider" not in config_data["embedding"]:
        raise ValueError("Missing required field: embedding.provider")

    # Validate Vector Store
    if "backend" not in config_data["vector_store"]:
        raise ValueError("Missing required field: vector_store.backend")
