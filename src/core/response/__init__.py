"""
Response 包：将检索命中组装成 API 返回结构（正文、引用、多模态内容）。

本包负责把 HybridSearchHit 转为前端/MCP 所需的 content + structuredContent，
包括 Markdown 正文、引用列表与图片等，便于上层统一返回格式而不散落拼接逻辑。
"""
from __future__ import annotations
