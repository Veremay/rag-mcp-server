import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


class FileIntegrityRegistry:
    """
    Manages file integrity checks to support incremental ingestion.

    Persists a record of processed file hashes to skip files that haven't changed.
    Currently uses a simple JSON file for storage.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize the registry.

        Args:
            storage_path: Path to the JSON file storing the registry.
                          If None, defaults to data/cache/ingestion_history.json
        """
        if storage_path:
            self.storage_path = Path(storage_path)
        else:
            # Default to a safe location relative to the project root or current working dir
            # Assuming running from project root
            self.storage_path = Path("data/cache/ingestion_history.json")

        self._registry: Dict[str, Any] = {}  # Map file_hash -> status (str) or metadata (dict)
        self._load()

    def compute_sha256(self, file_path: Path) -> str:
        """
        Compute the SHA256 hash of a file.

        Args:
            file_path: Path to the file.

        Returns:
            The hex digest of the SHA256 hash.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read in chunks to handle large files efficiently
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def should_skip(self, file_hash: str) -> bool:
        """
        Check if the file with the given hash has already been successfully processed.

        Args:
            file_hash: The SHA256 hash of the file.

        Returns:
            True if the file should be skipped, False otherwise.
        """
        val = self._registry.get(file_hash)
        if isinstance(val, dict):
            return val.get("status") == "success"
        return val == "success"

    def mark_success(self, file_hash: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Mark a file hash as successfully processed.

        Args:
            file_hash: The SHA256 hash of the file.
            metadata: Optional metadata to store with the record (e.g., file path).
        """
        if metadata:
            self._registry[file_hash] = {"status": "success", **metadata}
        else:
            self._registry[file_hash] = "success"
        self._save()

    def _load(self) -> None:
        if self.storage_path.exists():
            try:
                self._registry = json.loads(self.storage_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self._registry = {}

    def remove_record(self, file_path: str) -> bool:
        """
        Remove a record from the registry by file path.

        Args:
            file_path: Absolute path to the file.
        
        Returns:
            True if the record was removed, False if not found.
        """
        to_remove = []
        for h, val in self._registry.items():
            if isinstance(val, dict) and val.get("path") == str(file_path):
                to_remove.append(h)
        
        if not to_remove:
            # Fallback: if user passes hash instead of path?
            # Or if path is not stored (legacy).
            # We can't safely remove legacy records by path.
            return False
            
        for h in to_remove:
            del self._registry[h]
            
        self._save()
        return True

    def list_processed(self) -> List[Dict[str, Any]]:
        """List all processed files with their metadata."""
        results = []
        for h, val in self._registry.items():
            if isinstance(val, dict):
                results.append({"hash": h, **val})
            else:
                results.append({"hash": h, "status": val})
        return results

    def _save(self) -> None:
        """Save the registry to disk."""
        # Ensure directory exists
        if not self.storage_path.parent.exists():
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(self._registry, f, indent=2)
