"""验收：TXT/PDF 能出块；三种策略 JSON 字段齐全；ingest 不 import LLM。"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from src.common.types import new_block
from src.ingest.chunking import (
    STRATEGIES,
    chunk_fixed,
    chunk_hierarchical,
    chunk_semantic,
    chunks_to_dicts,
)
from src.ingest.parse_docs import parse_file
from src.ingest.parse_txt import parse_txt
from src.ingest.run_chunk import process_file, run_chunk

ROOT = Path(__file__).resolve().parents[2]
DEMO_TXT = ROOT / "shared" / "sample_docs" / "demo.txt"
INGEST_DIR = ROOT / "src" / "ingest"
REQUIRED_FIELDS = (
    "chunk_id",
    "text",
    "source",
    "page_num",
    "section_path",
    "strategy",
    "block_types",
    "stock_code",
    "year",
)
FORBIDDEN_IMPORTS = {"openai", "dashscope", "langchain", "faiss"}


def _text_block(text: str, section: str = "一、概述", page: int = 1) -> dict:
    return new_block("text", text, source="demo.txt", page_num=page, section_path=[section])


def _title_block(text: str) -> dict:
    return new_block("title", text, source="demo.txt", page_num=1, section_path=[text])


def _table_block(text: str, section: str = "五、主要财务数据") -> dict:
    return new_block("table", text, source="demo.txt", page_num=1, section_path=[section])


def _write_sample_pdf(path: Path) -> None:
    import pymupdf as fitz

    doc = fitz.open()
    page = doc.new_page()
    font = "china-ss"
    page.insert_text((72, 72), "示例股份有限公司", fontsize=16, fontname=font)
    page.insert_text((72, 110), "一、主营业务概述", fontsize=14, fontname=font)
    page.insert_text((72, 140), "本公司主要从事示例消费品的研发、生产与销售。", fontsize=11, fontname=font)
    page.insert_text((72, 180), "二、盈利能力", fontsize=14, fontname=font)
    page.insert_text((72, 210), "2023年毛利率为45.2%，净利润28.6亿元。", fontsize=11, fontname=font)
    page.insert_text((72, 250), "五、主要财务数据", fontsize=14, fontname=font)
    page.insert_text((72, 280), "营业收入 120.5亿元", fontsize=11, fontname=font)
    doc.save(path, deflate=True, garbage=4)
    doc.close()


def test_parse_txt_emits_title_text_and_table_blocks():
    blocks = parse_txt(DEMO_TXT)
    types = {block["block_type"] for block in blocks}
    titles = [block["content"] for block in blocks if block["block_type"] == "title"]
    assert "title" in types
    assert "text" in types
    assert "table" in types
    assert any(title.startswith("一、") for title in titles)
    assert all(block.get("source") == "demo.txt" for block in blocks)
    sibling_titles = [block for block in blocks if block["block_type"] == "title" and "、" in block["content"]]
    assert sibling_titles
    assert all(len(block["section_path"]) == 1 for block in sibling_titles)


def test_parse_file_dispatches_by_suffix(tmp_path: Path):
    txt_blocks = parse_file(DEMO_TXT)
    assert txt_blocks
    with pytest.raises(ValueError, match="unsupported"):
        parse_file(tmp_path / "note.md")


def test_parse_pdf_emits_blocks(tmp_path: Path):
    pdf_path = tmp_path / "demo.pdf"
    _write_sample_pdf(pdf_path)
    blocks = parse_file(pdf_path)
    assert blocks
    assert any(block["block_type"] == "title" for block in blocks)
    assert any(block["page_num"] == 1 for block in blocks)
    assert all(block["source"] == "demo.pdf" for block in blocks)


def test_chunk_fixed_keeps_overlap_and_whole_table():
    """overlap 让相邻块共享切点附近字符；表格不得被滑窗切开。"""
    table = "| 项目 | 金额 |\n| --- | --- |\n| 营收 | 120 |"
    blocks = [
        _text_block("A" * 40),
        _table_block(table),
        _text_block("B" * 40),
    ]
    chunks = chunk_fixed(blocks, size=30, overlap=10)
    texts = [chunk.text for chunk in chunks]
    assert table in texts
    windowed = [text for text in texts if text != table]
    assert len(windowed) >= 2
    assert windowed[0][20:30] == windowed[1][0:10]
    assert all(chunk.strategy == "fixed" for chunk in chunks)
    table_chunk = next(chunk for chunk in chunks if chunk.text == table)
    assert table_chunk.block_types == ["table"]


def test_chunk_semantic_splits_on_title_and_keeps_table():
    """标题强制切段；表格独立成块。"""
    table = "| 项目 | 金额 |\n| --- | --- |\n| 营收 | 120 |"
    blocks = [
        _title_block("一、主营业务概述"),
        _text_block("营收增长。", "一、主营业务概述"),
        _title_block("二、盈利能力"),
        _text_block("毛利率提升。", "二、盈利能力"),
        _table_block(table, "二、盈利能力"),
    ]
    chunks = chunk_semantic(blocks, max_chars=80)
    assert any("一、主营业务概述" in chunk.text for chunk in chunks)
    assert any("二、盈利能力" in chunk.text for chunk in chunks)
    assert any(chunk.text == table for chunk in chunks)
    assert all(chunk.strategy == "semantic" for chunk in chunks)
    table_chunk = next(chunk for chunk in chunks if chunk.text == table)
    assert table_chunk.block_types == ["table"]


def test_chunk_hierarchical_children_point_to_parent():
    """Small-to-Big：子块 parent_id 必须指向已写出的父块。"""
    blocks = [
        _title_block("一、主营业务概述"),
        _text_block("A" * 50, "一、主营业务概述"),
        _title_block("二、盈利能力"),
        _text_block("B" * 50, "二、盈利能力"),
    ]
    chunks = chunk_hierarchical(blocks, child_size=40, overlap=5)
    parents = [chunk for chunk in chunks if chunk.parent_id is None]
    children = [chunk for chunk in chunks if chunk.parent_id is not None]
    parent_ids = {chunk.chunk_id for chunk in parents}
    assert parents
    assert children
    assert all(chunk.parent_id in parent_ids for chunk in children)
    assert all(chunk.strategy == "hierarchical" for chunk in chunks)


def test_chunk_infers_stock_year_from_filename():
    blocks = [
        new_block(
            "text",
            "经营情况概述。",
            source="600519_2023_贵州茅台.pdf",
            page_num=1,
            section_path=["一、概述"],
        )
    ]
    chunk = chunk_fixed(blocks, size=500, overlap=50)[0]
    assert chunk.stock_code == "600519"
    assert chunk.year == "2023"
    assert chunk.block_types == ["text"]


def test_chunks_have_required_fields():
    blocks = parse_txt(DEMO_TXT)
    for name, fn in STRATEGIES.items():
        rows = chunks_to_dicts(fn(blocks))
        assert rows
        for row in rows:
            for field in REQUIRED_FIELDS:
                assert field in row
            assert row["strategy"] == name
            assert row["source"] == "demo.txt"
            assert row["stock_code"] == "600000"
            assert row["year"] == "2023"
            assert row["block_types"]


def test_process_file_writes_three_strategy_json(tmp_path: Path):
    written = process_file(DEMO_TXT, tmp_path / "chunks", tmp_path / "parsed")
    names = {path.name for path in written}
    assert names == {
        "demo_fixed.json",
        "demo_semantic.json",
        "demo_hierarchical.json",
    }
    parsed = json.loads((tmp_path / "parsed" / "demo.json").read_text(encoding="utf-8"))
    assert parsed and parsed[0]["block_type"] in {"title", "text", "table"}


def test_run_chunk_on_raw_dir(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "demo.txt").write_text(DEMO_TXT.read_text(encoding="utf-8"), encoding="utf-8")
    _write_sample_pdf(raw / "sample_report.pdf")
    written = run_chunk(raw, tmp_path / "chunks", tmp_path / "parsed")
    names = {path.name for path in written}
    assert "demo_semantic.json" in names
    assert "sample_report_fixed.json" in names
    assert "sample_report_hierarchical.json" in names


def test_ingest_modules_do_not_import_llm():
    """V1 闸门：解析分块链路禁止绑 LLM，才能单独单测。"""
    for py in INGEST_DIR.glob("*.py"):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert imported.isdisjoint(FORBIDDEN_IMPORTS), py.name
