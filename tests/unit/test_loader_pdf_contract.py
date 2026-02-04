import pytest
from pathlib import Path
import tempfile
import os
from src.libs.loader.base_loader import BaseLoader
from src.libs.loader.pdf_loader import PdfLoader
from src.ingestion.models import Document

class TestPdfLoader:
    
    @pytest.fixture
    def sample_pdf(self):
        """Create a dummy PDF file for testing."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(b"%PDF-1.4 header dummy content")
            file_path = Path(f.name)
        yield file_path
        if file_path.exists():
            os.unlink(file_path)

    def test_inheritance(self):
        """Verify PdfLoader inherits from BaseLoader."""
        loader = PdfLoader()
        assert isinstance(loader, BaseLoader)

    def test_load_returns_document(self, sample_pdf):
        """Verify load returns a Document with correct metadata."""
        loader = PdfLoader()
        doc = loader.load(sample_pdf)
        
        assert isinstance(doc, Document)
        assert doc.metadata["source_path"] == str(sample_pdf.absolute())
        assert doc.metadata["filename"] == sample_pdf.name
        assert doc.metadata["extension"] == ".pdf"
        # Text is empty for shell implementation
        assert doc.text == ""

    def test_file_not_found(self):
        """Verify FileNotFoundError is raised for non-existent file."""
        loader = PdfLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("non_existent_file.pdf")
