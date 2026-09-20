"""AST 白名单沙箱计算器。禁止 eval；拒绝导入与属性访问。"""

from __future__ import annotations

import ast
import operator as op

from src.common.config import CALC_MAX_EXPR_LEN
from src.common.types import ToolResult, fail, ok

_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": (
            "安全计算数学表达式，仅支持加减乘除与幂运算。"
            "财务差值、增长率、百分比必须调用本工具，禁止心算。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expr": {
                    "type": "string",
                    "description": "数学表达式，如 '(91.96 - 75.79)' 或 '(747 - 524) / 524 * 100'",
                },
            },
            "required": ["expr"],
        },
    },
}


def _eval(node: ast.AST) -> float | int:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(
        node.value, bool
    ):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left = _eval(node.left)
        right = _eval(node.right)
        if isinstance(node.op, ast.Pow) and (abs(left) > 1e6 or abs(right) > 20):
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported")


def calculator(expr: str) -> ToolResult:
    text = (expr or "").strip()
    if not text:
        return fail("expr 不能为空")
    if len(text) > CALC_MAX_EXPR_LEN:
        return fail(f"表达式过长（>{CALC_MAX_EXPR_LEN}）")
    try:
        tree = ast.parse(text, mode="eval")
        value = _eval(tree)
    except ZeroDivisionError:
        return fail("除以零")
    except Exception as e:
        return fail(str(e))
    if isinstance(value, float):
        value = round(value, 8)
    return ok(value)


def tool_result_text(result: ToolResult) -> str:
    if not result.get("ok"):
        return f"[错误] {result.get('error') or '未知错误'}"
    return str(result.get("data"))
