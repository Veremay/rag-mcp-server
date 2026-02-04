from typing import Dict, Type, Optional

from src.core.settings import Settings
from src.libs.splitter.base_splitter import BaseSplitter

def _try_load_recursive_splitter() -> Optional[Type[BaseSplitter]]:
    try:
        from src.libs.splitter.recursive_splitter import RecursiveSplitter
    except ModuleNotFoundError:
        return None
    return RecursiveSplitter

class SplitterFactory:
    """
    Factory for creating Splitter instances based on configuration.
    Supports dynamic registration of providers.
    """
    _registry: Dict[str, Type[BaseSplitter]] = {}

    _recursive_splitter = _try_load_recursive_splitter()
    if _recursive_splitter is not None:
        _registry["recursive"] = _recursive_splitter

    @classmethod
    def register(cls, provider: str, splitter_cls: Type[BaseSplitter]) -> None:
        """
        Register a new splitter provider class.

        Args:
            provider: The provider name (e.g., "recursive", "semantic").
            splitter_cls: The class implementing BaseSplitter.
        """
        cls._registry[provider.lower()] = splitter_cls

    @classmethod
    def create(cls, settings: Settings) -> BaseSplitter:
        """
        Create a Splitter instance based on the provided settings.

        Args:
            settings: The global application settings.

        Returns:
            An instance of BaseSplitter.

        Raises:
            ValueError: If the provider is not registered.
        """
        provider = settings.ingestion.splitter.provider.lower()
        
        if provider not in cls._registry:
            raise ValueError(
                f"Unknown splitter provider: '{provider}'. "
                f"Available providers: {list(cls._registry.keys())}"
            )
        
        splitter_cls = cls._registry[provider]
        
        return splitter_cls(settings)
