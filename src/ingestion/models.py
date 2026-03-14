"""
摄取数据模型：原始文档与分块的结构定义。

Document 表示加载后的单文档，Chunk 表示切分后的文本块并携带 doc_id、起止位置等，
便于流水线各阶段传递与追溯来源，且与向量库/BM25 的 record 结构对齐。
"""
import uuid
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


def generate_uuid() -> str:
    """Generate a UUID4 string. 用于 Document/Chunk 的默认 id，保证全局唯一避免冲突。"""
    return str(uuid.uuid4())


class Document(BaseModel):
    """
    Represents a raw document before splitting.

    Attributes:
        id: Unique identifier for the document. Defaults to UUID4.
        text: The content of the document.
        metadata: Arbitrary metadata associated with the document.

    切分前的单文档；metadata 可含 image_refs、source 等，供 loader 与后续 transform 使用。
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

    分块后供编码与落库；metadata 会携带 collection、section_path 等，与 VectorRecord 一致。
    """

    id: str = Field(default_factory=generate_uuid)
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    doc_id: Optional[str] = None
    start_char_idx: Optional[int] = None
    end_char_idx: Optional[int] = None
