from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

from src.ingestion.models import Document


class BaseLoader(ABC):
    """
    Abstract base class for document loaders.

    All loaders must implement the load method to return a Document object.
    """

    @abstractmethod
    def load(self, file_path: Union[str, Path]) -> Document:
        """
        Load a file and return a Document.

        Args:
            file_path: Path to the file to load.

        Returns:
            A Document object containing the text and metadata.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file format is not supported or parsing fails.
        """
        pass
