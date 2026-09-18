"""表格 wrap 行合并、跨页续表合并、semantic 整块。"""

from __future__ import annotations

from src.common.types import new_block
from src.ingest.chunking import chunk_semantic
from src.ingest.parse_pdf import (
    _merge_continued_table_blocks,
    _merge_wrapped_table_rows,
    table_to_markdown,
)


def test_merge_wrapped_rows_joins_label_value_label():
    rows = [
        ["主要会计数据", "2023年", "2022年", "同比增减", "2021年"],
        ["归属于上市公司股东的扣除非经常", "", "", "", ""],
        ["", "23,327,729,257.82", "19,994,943,929.15", "16.67%", "17,405,930,787.45"],
        ["性损益的净利润（元）", "", "", "", ""],
    ]
    merged = _merge_wrapped_table_rows(rows)
    assert len(merged) == 2
    assert merged[1][0] == "归属于上市公司股东的扣除非经常性损益的净利润（元）"
    assert "23,327,729,257.82" in merged[1]
    assert "16.67%" in merged[1]


def test_table_to_markdown_applies_wrap_merge():
    table = [
        ["项目", "金额", "备注"],
        ["营业收入合", "", ""],
        ["", "120.5", "—"],
        ["计（亿元）", "", ""],
    ]
    md = table_to_markdown(table)
    assert "营业收入合计（亿元）" in md
    assert "120.5" in md
    # 不应再残留拆开的半截标签行作为独立数据行
    assert md.count("营业收入合") == 1


def test_merge_continued_tables_across_pages_with_noise():
    header = (
        "| 项目 | 营业收入 |\n"
        "| --- | --- |"
    )
    body = "| 销售模式 | 61,731,839,992.96 |"
    blocks = [
        new_block("table", header, source="a.pdf", page_num=8, section_path=["五、财务"]),
        new_block(
            "title",
            "宜宾五粮液股份有限公司2021年年度报告",
            source="a.pdf",
            page_num=9,
            section_path=["五、财务"],
        ),
        new_block(
            "table",
            f"{header}\n{body}",
            source="a.pdf",
            page_num=9,
            section_path=["五、财务"],
        ),
    ]
    merged = _merge_continued_table_blocks(blocks)
    assert len(merged) == 1
    assert merged[0]["block_type"] == "table"
    assert "销售模式" in merged[0]["content"]
    assert "61,731,839,992.96" in merged[0]["content"]
    assert merged[0]["page_num"] == 8


def test_merge_continued_tables_skips_when_real_title_between():
    t1 = "| 项目 | 金额 |\n| --- | --- |\n| A | 1 |"
    t2 = "| 项目 | 金额 |\n| --- | --- |\n| B | 2 |"
    blocks = [
        new_block("table", t1, source="a.pdf", page_num=1, section_path=["一"]),
        new_block("title", "二、盈利能力分析", source="a.pdf", page_num=2, section_path=["二"]),
        new_block("table", t2, source="a.pdf", page_num=2, section_path=["二"]),
    ]
    merged = _merge_continued_table_blocks(blocks)
    assert len(merged) == 3
    assert [b["block_type"] for b in merged] == ["table", "title", "table"]


def test_chunk_semantic_keeps_merged_cross_page_table_as_one():
    header = "| 项目 | 营业收入 |\n| --- | --- |"
    body = "| 销售模式 | 100 |"
    blocks = _merge_continued_table_blocks(
        [
            new_block("table", header, source="a.pdf", page_num=8, section_path=["五"]),
            new_block(
                "title",
                "某某公司2021年年度报告",
                source="a.pdf",
                page_num=9,
                section_path=["五"],
            ),
            new_block(
                "table",
                f"{header}\n{body}",
                source="a.pdf",
                page_num=9,
                section_path=["五"],
            ),
        ]
    )
    chunks = chunk_semantic(blocks, max_chars=80)
    table_chunks = [c for c in chunks if "table" in c.block_types]
    assert len(table_chunks) == 1
    assert "销售模式" in table_chunks[0].text
    assert "100" in table_chunks[0].text
