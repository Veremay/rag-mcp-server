from typing import List, Optional, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter as LCRecursiveSplitter
from src.libs.splitter.base_splitter import BaseSplitter, TraceContext

class RecursiveSplitter(BaseSplitter):
    """
    Recursive splitter implementation wrapping langchain-text-splitters.
    """
    def __init__(self, settings: Any):
        """
        Initialize the recursive splitter with settings.
        
        Args:
            settings: Global settings object containing ingestion.splitter config
        """
        splitter_config = settings.ingestion.splitter
        self.chunk_size = splitter_config.chunk_size
        self.chunk_overlap = splitter_config.chunk_overlap
        
        # Initialize LangChain splitter
        # We use the standard separators list which is good for Markdown and Code
        self._splitter = LCRecursiveSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )

    def split_text(self, text: str, trace: Optional[TraceContext] = None, **kwargs: Any) -> List[str]:
        """
        Split text into chunks using recursive character splitting.
        
        Args:
            text: The text to split.
            trace: Optional trace context (unused).
            **kwargs: Additional arguments passed to LangChain splitter.
            
        Returns:
            List of text chunks.
        """
        return self._splitter.split_text(text)
