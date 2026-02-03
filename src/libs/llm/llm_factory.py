from src.core.settings import Settings
from src.libs.llm.base_llm import BaseLLM
from src.libs.llm.openai_llm import OpenAILLM
from src.libs.llm.azure_llm import AzureOpenAILLM
from src.libs.llm.deepseek_llm import DeepSeekLLM
from src.libs.llm.ollama_llm import OllamaLLM

class LLMFactory:
    """
    Factory class for creating LLM instances based on configuration.
    """
    
    @staticmethod
    def create(settings: Settings) -> BaseLLM:
        """
        Create an LLM instance based on the provided settings.
        
        Args:
            settings: Application settings containing LLM configuration
            
        Returns:
            An instance of a class implementing BaseLLM
            
        Raises:
            ValueError: If the configured provider is not supported or missing config.
        """
        provider = settings.llm.provider.lower()
        
        if provider == "openai":
            if not settings.llm.api_key:
                raise ValueError("OpenAI provider requires api_key")
            return OpenAILLM(
                api_key=settings.llm.api_key,
                model=settings.llm.model,
                base_url=settings.llm.base_url
            )
            
        elif provider == "azure":
            if not settings.llm.api_key:
                raise ValueError("Azure provider requires api_key")
            if not settings.llm.azure_endpoint:
                raise ValueError("Azure provider requires azure_endpoint")
            return AzureOpenAILLM(
                api_key=settings.llm.api_key,
                azure_endpoint=settings.llm.azure_endpoint,
                model=settings.llm.model
            )
            
        elif provider == "deepseek":
            if not settings.llm.api_key:
                raise ValueError("DeepSeek provider requires api_key")
            # base_url is optional for DeepSeekLLM as it has a default
            kwargs = {}
            if settings.llm.base_url:
                kwargs["base_url"] = settings.llm.base_url
                
            return DeepSeekLLM(
                api_key=settings.llm.api_key,
                model=settings.llm.model,
                **kwargs
            )

        elif provider == "ollama":
            # base_url is optional (defaults to localhost), api_key is optional
            kwargs = {}
            if settings.llm.base_url:
                kwargs["base_url"] = settings.llm.base_url
            if settings.llm.api_key:
                kwargs["api_key"] = settings.llm.api_key
                
            return OllamaLLM(
                model=settings.llm.model,
                **kwargs
            )
            
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")
