"""TXT Loader：按行识别年报常见标题 / Markdown 表 / 正文，产出 block 列表。

只负责结构化，不分块、不调 LLM。标题规则与 parse_pdf 共用，保证 TXT/PDF 下游分块行为一致。
"""

from __future__ import annotations

import re
from pathlib import Path

from src.common.types import new_block

# 年报目录体例：第一章 / 一、 / 1、 / 1.
CHAPTER_PATTERNS = [
    re.compile(r"^第[一二三四五六七八九十百]+[章节]"),
    re.compile(r"^[一二三四五六七八九十]、"),
    re.compile(r"^\d+、"),
    re.compile(r"^\d+\.\s"),
]


def is_title_line(line: str, fontsize: float | None = None, is_bold: bool = False) -> bool:
    """PDF 优先用字号/加粗；无字体信息时（TXT）退回章节编号正则。"""
    stripped = line.strip()
    if not stripped:
        return False
    if fontsize and fontsize >= 14:
        return True
    if is_bold and len(stripped) < 50:
        return True
    return any(pattern.match(stripped) for pattern in CHAPTER_PATTERNS)


def is_markdown_table_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2


def update_section_stack(stack: list[str], title: str) -> list[str]:
    """维护章节路径栈，供 Chunk.section_path 引用。

    有「第X章」时：章 → 节 → 一、 为三级；样例摘要没有章时，「一、」「二、」视为同级，避免错误嵌套。
    """
    if re.match(r"^第[一二三四五六七八九十]+章", title):
        return [title]
    if re.match(r"^第[一二三四五六七八九十]+节", title):
        return stack[:1] + [title]
    if re.match(r"^[一二三四五六七八九十]、", title) or re.match(r"^\d+、", title):
        if stack and re.match(r"^第[一二三四五六七八九十]+章", stack[0]):
            return stack[:2] + [title]
        return [title]
    return stack[:3] + [title]


def read_text(path: Path) -> str:
    """UTF-8 优先，兼容 Windows 下导出的 gb18030 年报摘录。"""
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def parse_txt(path: str | Path) -> list[dict]:
    path = Path(path)
    source = path.name
    lines = read_text(path).splitlines()

    blocks: list[dict] = []
    section_stack: list[str] = []
    para_lines: list[str] = []
    table_lines: list[str] = []

    def flush_para() -> None:
        if not para_lines:
            return
        blocks.append(
            new_block(
                "text",
                "\n".join(para_lines),
                source=source,
                page_num=1,
                section_path=section_stack,
            )
        )
        para_lines.clear()

    def flush_table() -> None:
        if not table_lines:
            return
        blocks.append(
            new_block(
                "table",
                "\n".join(table_lines),
                source=source,
                page_num=1,
                section_path=section_stack,
            )
        )
        table_lines.clear()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            flush_table()
            flush_para()
            continue
        if is_markdown_table_line(stripped):
            flush_para()
            table_lines.append(stripped)
            continue
        flush_table()
        if is_title_line(stripped):
            flush_para()
            section_stack = update_section_stack(section_stack, stripped)
            blocks.append(
                new_block(
                    "title",
                    stripped,
                    source=source,
                    page_num=1,
                    section_path=section_stack,
                )
            )
            continue
        para_lines.append(stripped)

    flush_table()
    flush_para()
    return blocks
