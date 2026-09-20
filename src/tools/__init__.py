"""V4+ 可调用业务后端：纯逻辑，不含 MCP/CLI/Prompt。V5 起经 registry 注册。"""

from __future__ import annotations

import json
from typing import Any, Callable

from src.common.types import ToolResult, fail
from src.tools.calculator import TOOL_SCHEMA as CALCULATOR_SCHEMA
from src.tools.calculator import calculator
from src.tools.calculator import tool_result_text as calculator_text
from src.tools.company_lookup import TOOL_SCHEMA as COMPANY_LOOKUP_SCHEMA
from src.tools.company_lookup import company_lookup
from src.tools.company_lookup import tool_result_text as company_lookup_text
from src.tools.financial_indicator import TOOL_SCHEMA as FINANCIAL_INDICATOR_SCHEMA
from src.tools.financial_indicator import financial_indicator
from src.tools.financial_indicator import tool_result_text as financial_indicator_text
from src.tools.rag_backend import rag_search
from src.tools.rag_backend import tool_result_text as rag_search_text
from src.tools.stock_price import TOOL_SCHEMA as STOCK_PRICE_SCHEMA
from src.tools.stock_price import stock_price
from src.tools.stock_price import tool_result_text as stock_price_text

RAG_SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "rag_search",
        "description": (
            "在 A 股年报语料库中语义检索原文段落，适合战略、风险因素、管理层讨论等定性内容，"
            "以及需要引用年报页码的财务表述。知识库仅含贵州茅台/五粮液/宁德时代/海康威视/中国平安，"
            "年份 2021–2025。query 尽量用短财务术语，公司与年份用 stock_code/year 过滤。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "检索问题。不要包含公司名和年份，例如 '毛利率'、'主要风险因素'",
                },
                "stock_code": {
                    "type": "string",
                    "description": "可选，6 位股票代码过滤，如 '600519'",
                },
                "year": {
                    "type": "string",
                    "description": "可选，年份过滤，如 '2023'",
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回段落数，默认 5",
                },
            },
            "required": ["query"],
        },
    },
}

TOOLS_MAP: dict[str, Callable[..., ToolResult]] = {
    "company_lookup": company_lookup,
    "rag_search": rag_search,
    "financial_indicator": financial_indicator,
    "stock_price": stock_price,
    "calculator": calculator,
}

TEXT_FNS: dict[str, Callable[[ToolResult], str]] = {
    "company_lookup": company_lookup_text,
    "rag_search": rag_search_text,
    "financial_indicator": financial_indicator_text,
    "stock_price": stock_price_text,
    "calculator": calculator_text,
}

TOOLS_SCHEMA = [
    COMPANY_LOOKUP_SCHEMA,
    RAG_SEARCH_SCHEMA,
    FINANCIAL_INDICATOR_SCHEMA,
    STOCK_PRICE_SCHEMA,
    CALCULATOR_SCHEMA,
]


def execute_tool(name: str, args: dict[str, Any] | None = None) -> tuple[ToolResult, str]:
    """执行已注册工具，返回 (ToolResult, 给 LLM 的 Observation 文本)。"""
    fn = TOOLS_MAP.get(name)
    if fn is None:
        result = fail(f"未知工具 '{name}'，可用：{list(TOOLS_MAP)}")
        return result, result["error"] or "未知工具"
    try:
        result = fn(**(args or {}))
    except TypeError as e:
        result = fail(f"参数错误：{e}")
    except Exception as e:
        result = fail(f"工具执行失败：{e}")
    text_fn = TEXT_FNS.get(name)
    if text_fn is not None:
        try:
            return result, text_fn(result)
        except Exception:
            pass
    return result, json.dumps(result, ensure_ascii=False)
