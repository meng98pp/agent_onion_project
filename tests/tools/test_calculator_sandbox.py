"""验收：AST 沙箱允许四则运算，拒绝导入与属性访问。"""

from __future__ import annotations

from src.tools.calculator import calculator


def test_arithmetic_ok():
    r = calculator("1 + 2 * 3")
    assert r["ok"] is True
    assert r["data"] == 7
    assert r["error"] is None


def test_div_and_paren():
    r = calculator("(91.96 - 75.79)")
    assert r["ok"] is True
    assert abs(float(r["data"]) - 16.17) < 1e-6


def test_reject_import():
    r = calculator("__import__('os')")
    assert r["ok"] is False
    assert r["data"] is None
    assert r["error"]


def test_reject_attribute_access():
    r = calculator("(1).__class__")
    assert r["ok"] is False
    assert r["data"] is None
    assert r["error"]


def test_reject_empty_and_overlong():
    assert calculator("")["ok"] is False
    assert calculator("1+" * 200 + "1")["ok"] is False
