from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict


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
    section_path: str = ""
    strategy: Strategy = "fixed"
    parent_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def ok(data: Any) -> ToolResult:
    return {"ok": True, "data": data, "error": None}


def fail(error: str) -> ToolResult:
    return {"ok": False, "data": None, "error": error}
