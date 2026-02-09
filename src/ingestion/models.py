import uuid
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


def generate_uuid() -> str:
    """Generate a UUID4 string."""
    return str(uuid.uuid4())


class Document(BaseModel):
    """
    Represents a raw document before splitting.

    Attributes:
        id: Unique identifier for the document. Defaults to UUID4.
        text: The content of the document.
        metadata: Arbitrary metadata associated with the document.
    """

    id: str = Field(default_factory=generate_uuid)
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """
    Represents a chunk of text derived from a Document.

    Attributes:
        id: Unique identifier for the chunk. Defaults to UUID4.
        text: The content of the chunk.
        metadata: Arbitrary metadata.
        doc_id: The ID of the parent document this chunk belongs to.
        start_char_idx: The starting character index in the original text (optional).
        end_char_idx: The ending character index in the original text (optional).
    """

    id: str = Field(default_factory=generate_uuid)
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    doc_id: Optional[str] = None
    start_char_idx: Optional[int] = None
    end_char_idx: Optional[int] = None
