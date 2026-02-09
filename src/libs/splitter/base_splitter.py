from abc import ABC, abstractmethod
from typing import Any, List, Optional

# Placeholder for TraceContext until Observability module is fully implemented
TraceContext = Any


class BaseSplitter(ABC):
    """
    Abstract base class for Document Splitters.
    All splitter implementations must inherit from this class.
    """

    @abstractmethod
    def split_text(
        self, text: str, trace: Optional[TraceContext] = None, **kwargs: Any
    ) -> List[str]:
        """
        Split text into chunks.

        Args:
            text: The text to split.
            trace: Optional trace context for observability.
            **kwargs: Additional splitter-specific arguments.

        Returns:
            List of text chunks.
        """
        pass
