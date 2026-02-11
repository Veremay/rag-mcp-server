from typing import Any, Dict

from src.core.settings import Settings, load_settings


class ConfigService:
    def __init__(self) -> None:
        self._settings = load_settings()

    def get_settings(self) -> Settings:
        return self._settings

    def get_component_info(self) -> Dict[str, Any]:
        s = self._settings
        return {
            "llm": {
                "provider": s.llm.provider,
                "model": s.llm.model,
            },
            "embedding": {
                "provider": s.embedding.provider,
                "model": s.embedding.model,
            },
            "vector_store": {
                "backend": s.vector_store.backend,
                "collection": s.vector_store.collection_name,
                "path": s.vector_store.persist_path,
            },
            "retrieval": {
                "sparse": s.retrieval.sparse_backend,
                "fusion": s.retrieval.fusion_algorithm,
                "top_k": s.retrieval.top_k_final,
            },
        }
