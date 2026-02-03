from typing import Dict, Type
from src.core.settings import Settings
from src.libs.llm.base_llm import BaseLLM

class LLMFactory:
    """
    Factory class for creating LLM instances based on configuration.
    """
    
    _registry: Dict[str, Type[BaseLLM]] = {}
    
    @classmethod
    def register(cls, provider_name: str, llm_class: Type[BaseLLM]):
        """
        Register a new LLM provider implementation.
        
        Args:
            provider_name: The name of the provider (e.g., 'azure', 'openai')
            llm_class: The class implementing BaseLLM
        """
        cls._registry[provider_name] = llm_class
        
    @classmethod
    def create(cls, settings: Settings) -> BaseLLM:
        """
        Create an LLM instance based on the provided settings.
        
        Args:
            settings: Application settings containing LLM configuration
            
        Returns:
            An instance of a class implementing BaseLLM
            
        Raises:
            ValueError: If the configured provider is not registered
        """
        provider = settings.llm.provider.lower()
        
        if provider not in cls._registry:
            raise ValueError(f"Unknown LLM provider: {provider}. Available: {list(cls._registry.keys())}")
            
        llm_class = cls._registry[provider]
        # We pass the full settings object to let the specific implementation 
        # extract what it needs (api_key, endpoint, model, etc.)
        return llm_class(settings.llm)
