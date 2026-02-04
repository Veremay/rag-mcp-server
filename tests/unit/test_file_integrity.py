import os
import tempfile
from pathlib import Path
import pytest
import json
from src.libs.loader.file_integrity import FileIntegrityRegistry

class TestFileIntegrityRegistry:
    
    @pytest.fixture
    def temp_file(self):
        """Create a temporary file for hashing tests."""
        with tempfile.NamedTemporaryFile(delete=False, mode="wb") as f:
            f.write(b"test content")
            file_path = Path(f.name)
        yield file_path
        if file_path.exists():
            os.unlink(file_path)

    @pytest.fixture
    def temp_registry_path(self):
        """Create a temporary path for the registry JSON."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            registry_path = Path(f.name)
        # Close and remove so the class can create it
        os.unlink(registry_path)
        yield registry_path
        if registry_path.exists():
            os.unlink(registry_path)

    def test_compute_sha256(self, temp_file):
        registry = FileIntegrityRegistry()
        hash1 = registry.compute_sha256(temp_file)
        
        # Verify deterministic
        hash2 = registry.compute_sha256(temp_file)
        assert hash1 == hash2
        
        # Verify correct hash for "test content"
        # python -c "import hashlib; print(hashlib.sha256(b'test content').hexdigest())"
        # 6ae8a75555209fd6c44157c0aed8016e763ff435a19cf186f76863140143ff72
        expected = "6ae8a75555209fd6c44157c0aed8016e763ff435a19cf186f76863140143ff72"
        assert hash1 == expected

    def test_should_skip_and_mark_success(self, temp_registry_path):
        registry = FileIntegrityRegistry(storage_path=temp_registry_path)
        test_hash = "abc123hash"
        
        # Initially should not skip
        assert registry.should_skip(test_hash) is False
        
        # Mark success
        registry.mark_success(test_hash)
        
        # Now should skip
        assert registry.should_skip(test_hash) is True

    def test_persistence(self, temp_registry_path):
        # Create one registry and save state
        registry1 = FileIntegrityRegistry(storage_path=temp_registry_path)
        test_hash = "persisted_hash"
        registry1.mark_success(test_hash)
        
        # Create new registry instance pointing to same file
        registry2 = FileIntegrityRegistry(storage_path=temp_registry_path)
        assert registry2.should_skip(test_hash) is True

    def test_default_path(self):
        """Test that default path is set correctly."""
        registry = FileIntegrityRegistry()
        assert registry.storage_path == Path("data/cache/ingestion_history.json")

    def test_missing_file_handling(self):
        registry = FileIntegrityRegistry()
        with pytest.raises(FileNotFoundError):
            registry.compute_sha256(Path("non_existent_file.txt"))
