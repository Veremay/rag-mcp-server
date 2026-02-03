from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import yaml
import os
from pathlib import Path

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
class IngestionSettings:
    splitter: SplitterSettings

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
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    if not config_data:
        raise ValueError("Configuration file is empty")

    validate_settings(config_data)
    
    return Settings(
        llm=LLMSettings(**config_data.get("llm", {})),
        embedding=EmbeddingSettings(**config_data.get("embedding", {})),
        vision_llm=VisionLLMSettings(**config_data.get("vision_llm", {})),
        vector_store=VectorStoreSettings(**config_data.get("vector_store", {})),
        retrieval=RetrievalSettings(**config_data.get("retrieval", {})),
        rerank=RerankSettings(**config_data.get("rerank", {})),
        evaluation=EvaluationSettings(**config_data.get("evaluation", {})),
        observability=ObservabilitySettings(**config_data.get("observability", {})),
    )

def validate_settings(config_data: Dict[str, Any]) -> None:
    """Validate critical configuration fields."""
    required_sections = [
        "llm", "embedding", "vision_llm", "vector_store", 
        "ingestion", "retrieval", "rerank", "evaluation", "observability"
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
