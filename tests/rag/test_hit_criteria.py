"""content_hit / doc_hit 判据单测。"""

from __future__ import annotations

from src.rag.hit_criteria import (
    content_hit_at_k,
    doc_hit_at_k,
    extract_number_signals,
    number_matches_text,
)


def test_doc_hit_company_name_only():
    hits = [
        {
            "text": "审计意见涉及贵州茅台关联方披露，无净利润数字。",
            "source": "600519_2023_贵州茅台.pdf",
        }
    ]
    assert doc_hit_at_k(hits, "茅台") is True
    assert doc_hit_at_k(hits, "五粮液") is False


def test_content_hit_false_positive_company_without_number():
    q = {
        "type": "precise_number",
        "question": "贵州茅台2023年归属于上市公司股东的净利润是多少？",
        "ground_truth": "贵州茅台2023年归属于上市公司股东的净利润为747.34亿元，同比增长19.16%。",
        "must_cite_source_substring": "茅台",
    }
    hits = [
        {
            "text": "贵州茅台审计报告。宁德时代净利润507.45亿元。中国平安净利润1266.07亿元。",
            "source": "600519_2023_贵州茅台.pdf",
        }
    ]
    assert doc_hit_at_k(hits, "茅台") is True
    hit, debug = content_hit_at_k(hits, q)
    assert hit is False
    assert debug["matched_signals"] == []


def test_content_hit_true_with_yi_yuan_in_text():
    q = {
        "type": "simple_fact",
        "question": "贵州茅台2023年的营业收入是多少？",
        "ground_truth": "贵州茅台2023年营业收入为1476.94亿元，同比增长18.04%。",
        "must_cite_source_substring": "茅台",
    }
    hits = [
        {
            "text": "2023 年度，财务报表所示营业收入发生 额为人民币14,769,360.50万元。贵州茅台",
            "source": "600519_2023_贵州茅台_贵州茅台2023年年度报告.pdf",
        }
    ]
    hit, debug = content_hit_at_k(hits, q)
    assert hit is True
    assert debug["matched_signals"]


def test_content_hit_true_with_direct_yi_number():
    q = {
        "type": "simple_fact",
        "question": "海康威视2023年的营业总收入是多少？",
        "ground_truth": "海康威视2023年营业总收入约893亿元。",
        "must_cite_source_substring": "海康",
    }
    hits = [
        {
            "text": "2023 年海康威视初步完成了智能物联战略的转型，公司实现营业总收入893.40 亿元",
            "source": "002415_2023_海康威视_2023年年度报告.pdf",
        }
    ]
    hit, _ = content_hit_at_k(hits, q)
    assert hit is True


def test_time_trend_needs_two_numbers():
    q = {
        "type": "time_trend",
        "question": "贵州茅台2021年至2023年营业收入的变化趋势如何？",
        "ground_truth": "贵州茅台营业收入：2021年约1061.9亿元、2022年约1241.0亿元、2023年约1476.9亿元。",
        "must_cite_source_substring": "茅台",
    }
    only_one = [
        {
            "text": "贵州茅台2023年营业收入约1476.9亿元。",
            "source": "茅台2023.pdf",
        }
    ]
    hit_one, _ = content_hit_at_k(only_one, q)
    assert hit_one is False

    two = [
        {
            "text": "贵州茅台营业收入2021年约1061.9亿元，2023年约1476.9亿元。",
            "source": "茅台.pdf",
        }
    ]
    hit_two, debug = content_hit_at_k(two, q)
    assert hit_two is True
    assert len(debug["matched_signals"]) >= 2


def test_should_refuse_always_false():
    q = {
        "type": "should_refuse",
        "question": "目前贵州茅台的股价是多少？",
        "ground_truth": "此题应拒绝回答。",
        "must_cite_source_substring": "",
    }
    hits = [{"text": "贵州茅台股价讨论", "source": "x.pdf"}]
    hit, _ = content_hit_at_k(hits, q)
    assert hit is False


def test_manual_must_cite_text_signals():
    q = {
        "type": "precise_number",
        "question": "测试",
        "ground_truth": "无数字说明",
        "must_cite_source_substring": "茅台",
        "must_cite_text_signals": ["747.34"],
    }
    hits = [{"text": "茅台净利润747.34亿元", "source": "a.pdf"}]
    hit, _ = content_hit_at_k(hits, q)
    assert hit is True


def test_extract_signals_yi_to_wan():
    signals = extract_number_signals("净利润为747.34亿元，同比增长19.16%。")
    kinds = {s["kind"] for s in signals}
    assert "yi" in kinds
    assert "wan" in kinds
    assert "pct" in kinds
    yi = next(s for s in signals if s["kind"] == "yi" and abs(s["value"] - 747.34) < 1e-6)
    assert number_matches_text(yi, "归母净利润747.34亿元")
