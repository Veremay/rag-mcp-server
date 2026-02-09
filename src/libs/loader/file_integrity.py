import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Optional, Set


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

        self._registry: Dict[str, str] = {}  # Map file_hash -> status (e.g., "success")
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
        return self._registry.get(file_hash) == "success"

    def mark_success(self, file_hash: str) -> None:
        """
        Mark a file hash as successfully processed.

        Args:
            file_hash: The SHA256 hash of the file.
        """
        self._registry[file_hash] = "success"
        self._save()

    def _load(self) -> None:
        """Load the registry from disk."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    self._registry = json.load(f)
            except (json.JSONDecodeError, OSError):
                # If file is corrupted or unreadable, start with empty registry
                self._registry = {}
        else:
            self._registry = {}

    def _save(self) -> None:
        """Save the registry to disk."""
        # Ensure directory exists
        if not self.storage_path.parent.exists():
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(self._registry, f, indent=2)
