from src.libs.llm.openai_llm import OpenAILLM


class DeepSeekLLM(OpenAILLM):
    """
    DeepSeek LLM implementation (OpenAI-compatible).
    Default base_url: https://api.deepseek.com
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.deepseek.com",
        **kwargs,
    ):
        super().__init__(api_key=api_key, model=model, base_url=base_url, **kwargs)
