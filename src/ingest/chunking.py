"""三种分块策略：只消费 parse_* 产出的 block，不读 PDF/TXT，不调用 LLM。

为何要分块：Embedding 有长度上限；检索要细粒度命中；Prompt 有 token 成本。
三种策略对照：
  fixed         定长滑窗 + overlap。实现最简单，但常切断句子。
  semantic      遇标题强制切开、表格独立成块。保章节完整性。
  hierarchical  父块=章节（给 LLM 上下文），子块=细粒度（给向量检索），即 Small-to-Big。
表格原则：Markdown 整表一块，避免行列被窗切开后无法对齐。
"""

from __future__ import annotations

import re
from dataclasses import asdict
from pathlib import Path

from src.common.types import Chunk, Strategy

# 年报文件名常见：600519_2023_贵州茅台_...pdf
_STOCK_YEAR_IN_NAME = re.compile(r"(?<!\d)(\d{6})[_-](20\d{2})")
_YEAR_IN_NAME = re.compile(r"(20\d{2})")
_STOCK_IN_TEXT = re.compile(r"股票代码\s*[:：]\s*(\d{6})")
_YEAR_IN_TEXT = re.compile(r"(?:报告期|报告年度)\s*[:：]\s*(20\d{2})")


def _source_name(block: dict) -> str:
    return Path(str(block.get("source") or "unknown")).name


def _section_str(section_path: list[str] | str | None) -> str:
    if isinstance(section_path, str):
        return section_path
    return " > ".join(section_path or [])


def _block_types(blocks: list[dict]) -> list[str]:
    seen: list[str] = []
    for block in blocks:
        btype = block.get("block_type")
        if btype and btype not in seen:
            seen.append(str(btype))
    return seen


def _infer_stock_year(blocks: list[dict], source: str) -> tuple[str, str]:
    """优先用 block 自带字段，否则从文件名 / 文首「股票代码」「报告期」推断。"""
    stock = next((str(b.get("stock_code") or "") for b in blocks if b.get("stock_code")), "")
    year = next((str(b.get("year") or "") for b in blocks if b.get("year")), "")
    stem = Path(source).stem
    named = _STOCK_YEAR_IN_NAME.search(stem)
    if named:
        stock = stock or named.group(1)
        year = year or named.group(2)
    if not year:
        found_year = _YEAR_IN_NAME.search(stem)
        if found_year:
            year = found_year.group(1)
    if not stock or not year:
        blob = "\n".join(str(b.get("content") or "") for b in blocks[:12])
        if not stock:
            found_stock = _STOCK_IN_TEXT.search(blob)
            stock = found_stock.group(1) if found_stock else stock
        if not year:
            found_year = _YEAR_IN_TEXT.search(blob)
            year = found_year.group(1) if found_year else year
    return stock, year


def _meta(block_or_blocks: dict | list[dict]) -> dict:
    """溯源取窗口内块：页码用第一块（跨页只是近似）；block_types 汇总去重。"""
    blocks = block_or_blocks if isinstance(block_or_blocks, list) else [block_or_blocks]
    first = blocks[0]
    source = _source_name(first)
    stock, year = _infer_stock_year(blocks, source)
    return {
        "source": source,
        "page_num": first.get("page_num"),
        "section_path": _section_str(first.get("section_path")),
        "block_types": _block_types(blocks),
        "stock_code": stock,
        "year": year,
    }


def _join_contents(blocks: list[dict]) -> str:
    parts = [str(b.get("content") or "").strip() for b in blocks]
    return "\n\n".join(part for part in parts if part)


def _slide(text: str, size: int, overlap: int) -> list[str]:
    """按字符滑窗。overlap 防止关键句恰好落在切点两侧各一半。"""
    if not text:
        return []
    if size <= 0:
        raise ValueError("size must be positive")
    # step=1 保证 overlap >= size 时仍能前进，避免死循环
    step = max(size - max(overlap, 0), 1)
    pieces: list[str] = []
    start = 0
    while start < len(text):
        pieces.append(text[start : start + size])
        if start + size >= len(text):
            break
        start += step
    return pieces


class _ChunkBuilder:
    """按策略生成稳定 chunk_id：{stem}_{strategy}_{0000}。"""

    def __init__(
        self,
        strategy: Strategy,
        stock_code: str = "",
        year: str = "",
    ) -> None:
        self.strategy = strategy
        self.stock_code = stock_code
        self.year = year
        self.idx = 0

    def add(
        self,
        text: str,
        meta: dict,
        *,
        parent_id: str | None = None,
    ) -> Chunk:
        source = meta["source"]
        stem = Path(source).stem or "doc"
        chunk = Chunk(
            chunk_id=f"{stem}_{self.strategy}_{self.idx:04d}",
            text=text,
            source=source,
            page_num=meta.get("page_num"),
            section_path=meta.get("section_path") or "",
            strategy=self.strategy,
            parent_id=parent_id,
            block_types=list(meta.get("block_types") or []),
            stock_code=str(meta.get("stock_code") or self.stock_code),
            year=str(meta.get("year") or self.year),
        )
        self.idx += 1
        return chunk


def _doc_stock_year(blocks: list[dict]) -> tuple[str, str]:
    if not blocks:
        return "", ""
    return _infer_stock_year(blocks, _source_name(blocks[0]))


def _windows_keep_tables(
    blocks: list[dict],
    size: int,
    overlap: int,
) -> list[tuple[str, dict]]:
    """正文走滑窗；碰到 table 先把缓冲文字切完，再把整张表单独留下。"""
    windows: list[tuple[str, dict]] = []
    buffer: list[dict] = []

    def flush_buffer() -> None:
        text = _join_contents(buffer)
        if text:
            meta = _meta(buffer)
            windows.extend((piece, meta) for piece in _slide(text, size, overlap))
        buffer.clear()

    for block in blocks:
        if block.get("block_type") == "table":
            flush_buffer()
            content = str(block.get("content") or "").strip()
            if content:
                windows.append((content, _meta(block)))
            continue
        buffer.append(block)
    flush_buffer()
    return windows


def chunk_fixed(
    blocks: list[dict],
    size: int = 500,
    overlap: int = 50,
) -> list[Chunk]:
    """基线策略：块大小可预测，但会截断句子；表格仍整块保留。"""
    builder = _ChunkBuilder("fixed", *_doc_stock_year(blocks))
    return [builder.add(text, meta) for text, meta in _windows_keep_tables(blocks, size, overlap)]


def chunk_semantic(blocks: list[dict], max_chars: int = 800) -> list[Chunk]:
    """按结构切：title 开启新块（自身当上下文前缀），table 独立，text 累积到阈值。"""
    builder = _ChunkBuilder("semantic", *_doc_stock_year(blocks))
    chunks: list[Chunk] = []
    buffer: list[dict] = []
    buffer_len = 0

    def flush() -> None:
        nonlocal buffer_len
        text = _join_contents(buffer)
        if text:
            chunks.append(builder.add(text, _meta(buffer)))
        buffer.clear()
        buffer_len = 0

    for block in blocks:
        content = str(block.get("content") or "")
        btype = block.get("block_type")
        if btype == "title":
            flush()
            buffer.append(block)
            buffer_len = len(content)
            continue
        if btype == "table":
            flush()
            if content.strip():
                chunks.append(builder.add(content.strip(), _meta(block)))
            continue
        if buffer and buffer_len + len(content) > max_chars:
            flush()
        buffer.append(block)
        buffer_len += len(content)
    flush()
    return chunks


def _group_by_section(blocks: list[dict]) -> list[tuple[str, list[dict]]]:
    """连续相同 section_path 归为一章，作为 hierarchical 的父块边界。"""
    groups: list[tuple[str, list[dict]]] = []
    current_key: str | None = None
    current: list[dict] = []
    for block in blocks:
        key = _section_str(block.get("section_path"))
        if current_key is None or key == current_key:
            current_key = key
            current.append(block)
            continue
        groups.append((current_key, current))
        current_key = key
        current = [block]
    if current_key is not None:
        groups.append((current_key, current))
    return groups


def chunk_hierarchical(
    blocks: list[dict],
    child_size: int = 400,
    overlap: int = 50,
) -> list[Chunk]:
    """Small-to-Big：检索打中子块，把带 parent_id 的父块（整章）送给 LLM。"""
    builder = _ChunkBuilder("hierarchical", *_doc_stock_year(blocks))
    chunks: list[Chunk] = []
    for _, group in _group_by_section(blocks):
        parent_text = _join_contents(group)
        if not parent_text:
            continue
        parent = builder.add(parent_text, _meta(group))
        chunks.append(parent)
        for text, meta in _windows_keep_tables(group, child_size, overlap):
            chunks.append(builder.add(text, meta, parent_id=parent.chunk_id))
    return chunks


STRATEGIES = {
    "fixed": chunk_fixed,
    "semantic": chunk_semantic,
    "hierarchical": chunk_hierarchical,
}


def chunks_to_dicts(chunks: list[Chunk]) -> list[dict]:
    """落盘用：契约字段始终写出；空 extra 省略以免干扰对照。"""
    rows = []
    for chunk in chunks:
        row = asdict(chunk)
        extra = row.pop("extra", None)
        if extra:
            row["extra"] = extra
        rows.append(row)
    return rows
