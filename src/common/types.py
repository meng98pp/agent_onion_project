"""跨版本契约：ingest / rag / tools 只依赖这里的类型，不互相掏实现。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

BlockType = Literal["text", "table", "title"]


class ToolResult(TypedDict):
    ok: bool
    data: Any
    error: str | None


Strategy = Literal["fixed", "semantic", "hierarchical"]


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str
    page_num: int | None = None
    section_path: str = ""  # 章节路径，如 "第三章 > 一、经营情况"
    strategy: Strategy = "fixed"
    parent_id: str | None = None  # 仅 hierarchical 子块指向父块
    block_types: list[str] = field(default_factory=list)  # 本块含 text/table/title
    stock_code: str = ""
    year: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def new_block(
    block_type: BlockType,
    content: str,
    *,
    source: str,
    page_num: int | None = None,
    section_path: list[str] | None = None,
    is_ocr: bool = False,
) -> dict[str, Any]:
    """parse_* 统一构造 block，避免字段名在 TXT/PDF 两侧漂移。"""
    return {
        "block_type": block_type,
        "content": content,
        "page_num": page_num,
        "section_path": list(section_path or []),
        "source": source,
        "is_ocr": is_ocr,
    }


def ok(data: Any) -> ToolResult:
    return {"ok": True, "data": data, "error": None}


def fail(error: str) -> ToolResult:
    return {"ok": False, "data": None, "error": error}
