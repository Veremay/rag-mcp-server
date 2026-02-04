from pathlib import Path
from typing import Union
from src.ingestion.models import Document
from src.libs.loader.base_loader import BaseLoader

class PdfLoader(BaseLoader):
    """
    Loader for PDF documents.
    
    Currently a shell implementation.
    """
    
    def load(self, file_path: Union[str, Path]) -> Document:
        """
        Load a PDF file and return a Document.
        
        Args:
            file_path: Path to the PDF file.
            
        Returns:
            A Document object.
            
        Raises:
            FileNotFoundError: If the file does not exist.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
            
        # TODO: Implement actual PDF parsing (e.g., using MarkItDown or PyPDF2)
        # For now, return a placeholder document as per C3 "Shell" requirement.
        
        return Document(
            text="",  # Placeholder text
            metadata={
                "source_path": str(path.absolute()),
                "filename": path.name,
                "extension": path.suffix.lower()
            }
        )
