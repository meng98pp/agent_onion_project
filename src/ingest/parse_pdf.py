"""PDF Loader：文字用 PyMuPDF（带字号/加粗），表格用 pdfplumber 转 Markdown。

V1 不做 OCR。parse_* 只产出 block；单页失败跳过，避免一本坏 PDF 停掉整批。
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

from src.common.types import new_block
from src.ingest.parse_txt import is_title_line, update_section_stack

# 页眉页脚：公司名+年度报告、孤立页码、— 38 —
NOISE_PATTERNS = [
    re.compile(r"^.{1,40}年度报告\s*$"),
    re.compile(r"^\d+\s*$"),
    re.compile(r"^—\s*\d+\s*—$"),
]

ProgressFn = Callable[[int, int], None]


def is_noise_line(line: str) -> bool:
    line = line.strip()
    if len(line) < 2:
        return True
    return any(pattern.match(line) for pattern in NOISE_PATTERNS)


def _is_label_only_row(row: list[str]) -> bool:
    """仅首列有字、其余空：常见于单元格内换行被拆成多行。"""
    if not row or not str(row[0]).strip():
        return False
    return all(not str(c).strip() for c in row[1:])


def _is_value_continuation_row(row: list[str]) -> bool:
    """首列空、后列有数：换行标签对应的数值行。

    两列表只需 1 个数值格；更宽的表要求后列「至少半数」非空，避免误吞空行。
    """
    if not row or str(row[0]).strip():
        return False
    rest = row[1:]
    if not rest:
        return False
    nonempty = sum(1 for c in rest if str(c).strip())
    need = 1 if len(rest) <= 2 else max(2, (len(rest) + 1) // 2)
    return nonempty >= need


def _merge_wrapped_table_rows(rows: list[list[str]]) -> list[list[str]]:
    """合并 pdfplumber 因单元格换行拆出的标签行与数值行。

    典型模式：
      | 归属于上市公司股东的扣除非经常 |  |  |  |
      |  | 23,327... | 19,994... | 16.67% |
      | 性损益的净利润（元） |  |  |  |
    → 一行完整指标 + 数值。
    """
    if not rows:
        return rows
    merged: list[list[str]] = []
    i = 0
    while i < len(rows):
        row = list(rows[i])
        if (
            _is_label_only_row(row)
            and i + 1 < len(rows)
            and _is_value_continuation_row(rows[i + 1])
        ):
            label = str(row[0]).strip()
            values = list(rows[i + 1])
            consumed = 2
            if i + 2 < len(rows) and _is_label_only_row(rows[i + 2]):
                label = f"{label}{str(rows[i + 2][0]).strip()}"
                consumed = 3
            width = max(len(values), len(row), 1)
            values = values + [""] * (width - len(values))
            values[0] = label
            merged.append(values)
            i += consumed
            continue
        merged.append(row)
        i += 1
    return merged


def table_to_markdown(table: list[list]) -> str:
    """行列转 Markdown，方便后续整表进一个 chunk、也方便 LLM 读。"""
    if not table:
        return ""
    rows = [
        [str(cell or "").replace("\n", " ").strip() for cell in row]
        for row in table
    ]
    rows = _merge_wrapped_table_rows(rows)
    if not rows:
        return ""
    header = rows[0]
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in rows[1:]:
        padded = row + [""] * max(0, len(header) - len(row))
        lines.append("| " + " | ".join(padded[: len(header)]) + " |")
    return "\n".join(lines)


def _markdown_column_count(md: str) -> int:
    for line in (md or "").splitlines():
        line = line.strip()
        if line.startswith("|") and "---" not in line:
            # | a | b | → 两侧空段不算列
            parts = [p for p in line.split("|")]
            cells = parts[1:-1] if len(parts) >= 3 else parts
            return len(cells)
    return 0


def _strip_markdown_header(md: str) -> str:
    """去掉续表重复的表头与分隔行，只保留数据行。"""
    lines = [ln for ln in (md or "").splitlines() if ln.strip()]
    if len(lines) >= 2 and "---" in lines[1]:
        return "\n".join(lines[2:])
    return "\n".join(lines)


def _is_noise_block(block: dict) -> bool:
    """跨页表之间夹着的页眉/页码等，可丢弃后仍合并续表。"""
    content = str(block.get("content") or "").strip()
    if not content:
        return True
    if is_noise_line(content):
        return True
    btype = block.get("block_type")
    # 页眉常被识别成 title：含「年度报告」、孤立页码、或「9 / 143」
    if btype == "title" and len(content) <= 80:
        if "年度报告" in content:
            return True
        if re.fullmatch(r"\d+", content):
            return True
        if re.fullmatch(r"\d+\s*/\s*\d+", content):
            return True
    return False


def _merge_continued_table_blocks(blocks: list[dict]) -> list[dict]:
    """合并跨页续表：相邻 table（中间可夹噪声）且列数一致则拼成一块。"""
    if not blocks:
        return blocks
    merged: list[dict] = []
    i = 0
    while i < len(blocks):
        block = blocks[i]
        if block.get("block_type") != "table":
            merged.append(block)
            i += 1
            continue

        current = dict(block)
        current_content = str(current.get("content") or "")
        cols = _markdown_column_count(current_content)
        j = i + 1
        while j < len(blocks):
            # 跳过噪声夹层，记录区间
            k = j
            while k < len(blocks) and _is_noise_block(blocks[k]):
                k += 1
            if k >= len(blocks) or blocks[k].get("block_type") != "table":
                break
            nxt = blocks[k]
            page_gap = abs(
                int(nxt.get("page_num") or 0) - int(current.get("page_num") or 0)
            )
            nxt_cols = _markdown_column_count(str(nxt.get("content") or ""))
            if page_gap > 1 or cols == 0 or nxt_cols != cols:
                break
            body = _strip_markdown_header(str(nxt.get("content") or ""))
            if body.strip():
                current_content = current_content.rstrip() + "\n" + body.strip()
                current["content"] = current_content
            # page_num 保留首块；噪声夹层丢弃
            j = k + 1
        merged.append(current)
        i = j if j > i else i + 1
    return merged


def _line_in_table(line_bbox: tuple[float, ...], table_bboxes: list[tuple]) -> bool:
    """跳过表格区域内的文字，避免同一张表既当 table 又当 text。"""
    if not table_bboxes or not line_bbox or len(line_bbox) < 4:
        return False
    cx = (line_bbox[0] + line_bbox[2]) / 2
    cy = (line_bbox[1] + line_bbox[3]) / 2
    for bbox in table_bboxes:
        x0, y0, x1, y1 = bbox[:4]
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            return True
    return False


def _collect_table_items(plumb_page, page_num: int) -> tuple[list[dict], list[tuple]]:
    items: list[dict] = []
    table_bboxes: list[tuple] = []
    if plumb_page is None:
        return items, table_bboxes
    try:
        tables = list(plumb_page.find_tables() or [])
    except Exception:
        return items, table_bboxes
    for table_obj in tables:
        try:
            bbox = tuple(table_obj.bbox)
            table_bboxes.append(bbox)
            markdown = table_to_markdown(table_obj.extract())
        except Exception:
            continue
        if not markdown:
            continue
        items.append(
            {
                "kind": "table",
                "content": markdown,
                "page_num": page_num,
                "y": float(bbox[1]),
            }
        )
    return items, table_bboxes


def _collect_line_items(fitz_page, page_num: int, table_bboxes: list[tuple]) -> list[dict]:
    import pymupdf as fitz

    items: list[dict] = []
    # dict 模式才能拿到每个 span 的 size / font，用于猜标题
    page_dict = fitz_page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:  # 0=文字，1=图片
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            line_text = "".join(span["text"] for span in spans).strip()
            if not line_text or is_noise_line(line_text):
                continue
            bbox = tuple(line.get("bbox") or ())
            if _line_in_table(bbox, table_bboxes):
                continue
            fontsize = spans[0].get("size", 0) if spans else 0
            is_bold = any("Bold" in span.get("font", "") for span in spans)
            items.append(
                {
                    "kind": "line",
                    "content": line_text,
                    "page_num": page_num,
                    "y": float(bbox[1]) if len(bbox) >= 2 else 0.0,
                    "fontsize": fontsize,
                    "is_bold": is_bold,
                }
            )
    return items


def _parse_page(fitz_page, plumb_page, human_page: int) -> list[dict]:
    table_items, table_bboxes = _collect_table_items(plumb_page, human_page)
    line_items = _collect_line_items(fitz_page, human_page, table_bboxes)
    page_items = table_items + line_items
    # 表格先抽、文字后抽，按纵坐标排回阅读顺序，section_path 才跟得上
    page_items.sort(key=lambda item: (item["y"], 0 if item["kind"] == "line" else 1))
    return page_items


def _items_to_blocks(items: list[dict], source: str) -> list[dict]:
    blocks: list[dict] = []
    section_stack: list[str] = []
    para_lines: list[str] = []
    para_page = 1

    def flush_para() -> None:
        if not para_lines:
            return
        blocks.append(
            new_block(
                "text",
                "\n".join(para_lines),
                source=source,
                page_num=para_page,
                section_path=section_stack,
            )
        )
        para_lines.clear()

    for item in items:
        page_num = item["page_num"]
        if item["kind"] == "table":
            flush_para()
            blocks.append(
                new_block(
                    "table",
                    item["content"],
                    source=source,
                    page_num=page_num,
                    section_path=section_stack,
                )
            )
            continue
        line_text = item["content"]
        if is_title_line(line_text, item.get("fontsize"), bool(item.get("is_bold"))):
            flush_para()
            section_stack = update_section_stack(section_stack, line_text)
            blocks.append(
                new_block(
                    "title",
                    line_text,
                    source=source,
                    page_num=page_num,
                    section_path=section_stack,
                )
            )
            continue
        if not para_lines:
            para_page = page_num
        para_lines.append(line_text)
    flush_para()
    return blocks


def _open_pdfs(path: Path):
    import pdfplumber
    import pymupdf as fitz

    fitz_doc = fitz.open(str(path))
    if getattr(fitz_doc, "needs_pass", False):
        fitz_doc.authenticate("")
    try:
        plumber_doc = pdfplumber.open(path)
    except Exception as exc:
        print(f"  pdfplumber 打开失败，仅提取文字: {exc}", flush=True)
        plumber_doc = None
    return fitz_doc, plumber_doc


def parse_pdf(
    path: str | Path,
    progress: ProgressFn | None = None,
) -> list[dict]:
    path = Path(path)
    source = path.name
    items: list[dict] = []
    fitz_doc, plumber_doc = _open_pdfs(path)
    try:
        n_pages = len(fitz_doc)
        n_plumb = len(plumber_doc.pages) if plumber_doc is not None else 0
        for page_num in range(n_pages):
            if progress:
                progress(page_num + 1, n_pages)
            plumb_page = None
            if plumber_doc is not None and page_num < n_plumb:
                plumb_page = plumber_doc.pages[page_num]
            try:
                items.extend(_parse_page(fitz_doc[page_num], plumb_page, page_num + 1))
            except Exception as exc:
                print(f"  skip page {page_num + 1} of {source}: {exc}", flush=True)
    finally:
        if plumber_doc is not None:
            plumber_doc.close()
        fitz_doc.close()
    return _merge_continued_table_blocks(_items_to_blocks(items, source))
