"""Hit@K 判据：doc_hit（公司名）与 content_hit（公司名 + 金标数字信号）。"""

from __future__ import annotations

import re
from typing import Any

_NUM_RE = re.compile(r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+|\d+)")
_UNIT_YI = re.compile(r"([\d,.]+)\s*亿\s*元?")
_UNIT_WAN = re.compile(r"([\d,.]+)\s*万\s*元?")
_PCT_RE = re.compile(r"([\d,.]+)\s*%")

# cross_doc 无数字时的指标词兜底
_METRIC_KEYWORDS = (
    "毛利率",
    "净利率",
    "营业收入",
    "净利润",
    "研发",
    "风险",
    "每股收益",
    "保险业务收入",
    "营业总收入",
)


def _parse_raw_number(s: str) -> float | None:
    s = (s or "").strip().replace(",", "").replace("，", "")
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def extract_number_signals(ground_truth: str) -> list[dict[str, Any]]:
    """从 ground_truth 抽取数字信号（含单位换算候选）。"""
    gt = ground_truth or ""
    approx = "约" in gt
    signals: list[dict[str, Any]] = []
    seen: set[tuple[float, str]] = set()

    def add(value: float, kind: str) -> None:
        key = (round(value, 6), kind)
        if key in seen or value == 0:
            return
        seen.add(key)
        signals.append({"value": value, "kind": kind, "approx": approx})

    for m in _UNIT_YI.finditer(gt):
        v = _parse_raw_number(m.group(1))
        if v is None:
            continue
        add(v, "yi")
        add(v * 10_000, "wan")  # 亿元 → 万元
        add(v * 1e8, "yuan")  # 亿元 → 元

    for m in _UNIT_WAN.finditer(gt):
        v = _parse_raw_number(m.group(1))
        if v is None:
            continue
        add(v, "wan")
        add(v / 10_000, "yi")
        add(v * 10_000, "yuan")

    for m in _PCT_RE.finditer(gt):
        v = _parse_raw_number(m.group(1))
        if v is not None:
            add(v, "pct")

    # 无单位裸数字（跳过已被单位捕获的片段附近重复可接受）
    for m in _NUM_RE.finditer(gt):
        v = _parse_raw_number(m.group(1))
        if v is None:
            continue
        # 跳过过小的年份类整数（1900–2100）单独作为唯一信号意义不大，仍保留供匹配
        add(v, "raw")

    return signals


def _numbers_in_text(text: str) -> list[float]:
    out: list[float] = []
    for m in _NUM_RE.finditer(text or ""):
        v = _parse_raw_number(m.group(1))
        if v is not None:
            out.append(v)
    return out


def number_matches_text(signal: dict[str, Any], text: str) -> bool:
    """判断信号数字是否以等价形式出现在 text 中。"""
    target = float(signal["value"])
    kind = signal.get("kind") or "raw"
    approx = bool(signal.get("approx"))
    rel_tol = 0.02 if approx else 0.01
    abs_tol = max(1e-6, abs(target) * rel_tol)

    candidates = _numbers_in_text(text)
    if kind == "pct":
        # 百分比：text 中同值，或带 % 的同值
        for n in candidates:
            if abs(n - target) <= abs_tol:
                return True
        return False

    for n in candidates:
        if abs(n - target) <= abs_tol:
            return True
        # 文本里是「万元」口径而信号是亿元数值等，已在 extract 时展开多候选
    return False


def doc_hit_at_k(hits: list[dict[str, Any]], must_substr: str) -> bool:
    """TopK 的 text 或 source 是否包含必须子串（旧判据）。"""
    needle = (must_substr or "").strip()
    if not needle:
        return False
    for h in hits:
        blob = f"{h.get('text') or ''} {h.get('source') or ''}"
        if needle in blob:
            return True
    return False


def _hit_blob(h: dict[str, Any]) -> str:
    return f"{h.get('text') or ''} {h.get('source') or ''}"


def _company_in_hit(h: dict[str, Any], must_substr: str) -> bool:
    needle = (must_substr or "").strip()
    if not needle:
        return False
    return needle in _hit_blob(h)


def _canonical_signal_key(sig: dict[str, Any]) -> float:
    """同一金额的亿元/万元/元换算归一到可比键，避免 time_trend 假阳性。"""
    v = float(sig["value"])
    kind = sig.get("kind") or "raw"
    if kind == "wan":
        return round(v / 10_000.0, 4)
    if kind == "yuan":
        return round(v / 1e8, 4)
    return round(v, 4)


def _matched_signals_in_hit(
    h: dict[str, Any],
    signals: list[dict[str, Any]],
) -> list[float]:
    text = str(h.get("text") or "")
    matched: list[float] = []
    seen: set[float] = set()
    for sig in signals:
        if number_matches_text(sig, text):
            key = _canonical_signal_key(sig)
            if key not in seen:
                seen.add(key)
                matched.append(key)
    return matched


def _keywords_from_question(question: str) -> list[str]:
    q = question or ""
    return [kw for kw in _METRIC_KEYWORDS if kw in q]


def content_hit_at_k(
    hits: list[dict[str, Any]],
    question: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    """严格 content hit：公司子串 + 金标数字（或关键词兜底）。

    返回 (hit, debug)。
    """
    qtype = str(question.get("type") or "")
    must = str(question.get("must_cite_source_substring") or "")
    gt = str(question.get("ground_truth") or "")
    qtext = str(question.get("question") or "")

    debug: dict[str, Any] = {
        "matched_signals": [],
        "required_signal_count": 1,
        "mode": qtype or "default",
    }

    if qtype == "should_refuse" or not must.strip():
        debug["mode"] = "should_refuse_or_empty_must"
        return False, debug

    manual = question.get("must_cite_text_signals")
    if isinstance(manual, list) and manual:
        signals = []
        for item in manual:
            v = _parse_raw_number(str(item))
            if v is not None:
                signals.append({"value": v, "kind": "raw", "approx": "约" in gt})
    else:
        signals = extract_number_signals(gt)

    # 过滤过弱信号：纯四位年份（1900–2100）不当作唯一证据，但仍可计入多信号
    strong_signals = [
        s
        for s in signals
        if not (
            s.get("kind") in {"raw", "yi", "wan", "yuan"}
            and 1900 <= s["value"] <= 2100
            and float(s["value"]).is_integer()
        )
    ]
    if not strong_signals:
        strong_signals = signals

    need = 2 if qtype == "time_trend" else 1
    debug["required_signal_count"] = need
    keywords = _keywords_from_question(qtext)

    for h in hits:
        if not _company_in_hit(h, must):
            continue
        matched = _matched_signals_in_hit(h, strong_signals)
        if len(matched) >= need:
            debug["matched_signals"] = matched[:8]
            return True, debug
        # cross_doc：无足够数字时，指标关键词 + 公司名
        if qtype == "cross_doc_compare" and keywords:
            text = str(h.get("text") or "")
            if any(kw in text for kw in keywords):
                debug["matched_signals"] = matched
                debug["matched_keywords"] = [kw for kw in keywords if kw in text]
                return True, debug

    # 数字可分散在多个 hit：同一公司下按规范键累计
    if need >= 2 or (strong_signals and qtype != "cross_doc_compare"):
        all_matched: list[float] = []
        seen: set[float] = set()
        for h in hits:
            if not _company_in_hit(h, must):
                continue
            for v in _matched_signals_in_hit(h, strong_signals):
                if v not in seen:
                    seen.add(v)
                    all_matched.append(v)
        if len(all_matched) >= need:
            debug["matched_signals"] = all_matched[:8]
            debug["mode"] = f"{qtype}_multi_hit"
            return True, debug

    return False, debug
