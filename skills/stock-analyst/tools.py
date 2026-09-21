"""stock-analyst 技能自带工具。只包装 src.tools，不写股票算法。"""

from __future__ import annotations

from src.tools.calculator import calculator
from src.tools.company_lookup import company_lookup
from src.tools.financial_indicator import financial_indicator
from src.tools.rag_backend import rag_search
from src.tools.stock_price import stock_price

TOOLS = [
    {
        "name": "company_lookup",
        "description": (
            "将公司中文名称转换为 A 股股票代码；"
            "在调用 financial_indicator 或 stock_price 前必须先用此工具获取代码。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "公司中文名，如'贵州茅台'"},
            },
            "required": ["name"],
        },
        "handler": company_lookup,
    },
    {
        "name": "rag_search",
        "description": (
            "在 A 股年报语料库中语义检索原文段落，适合战略、风险因素、管理层讨论等定性内容。"
            "知识库仅含贵州茅台/五粮液/宁德时代/海康威视/中国平安，年份 2021–2025。"
            "query 尽量用短财务术语，公司与年份用 stock_code/year 过滤。"
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
        "handler": rag_search,
    },
    {
        "name": "financial_indicator",
        "description": (
            "获取 A 股近 3 年关键财务指标（营收/净利润/毛利率/ROE/资产负债率等），"
            "适合跨年对比或与年报交叉验证。symbol 必须是 6 位股票代码。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "股票代码，如'600519'"},
            },
            "required": ["symbol"],
        },
        "handler": financial_indicator,
    },
    {
        "name": "stock_price",
        "description": "获取 A 股历史股价及区间涨跌幅，日期格式为 YYYYMMDD。",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "股票代码，如'600519'"},
                "start_date": {
                    "type": "string",
                    "description": "起始日期 YYYYMMDD，如'20230101'",
                },
                "end_date": {
                    "type": "string",
                    "description": "结束日期 YYYYMMDD，如'20231231'",
                },
            },
            "required": ["symbol", "start_date", "end_date"],
        },
        "handler": stock_price,
    },
    {
        "name": "calculator",
        "description": (
            "安全计算数学表达式；在已取到财务数字后计算增长率、差值、百分比时使用。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expr": {
                    "type": "string",
                    "description": "数学表达式，如 '(747 - 524) / 524 * 100'",
                },
            },
            "required": ["expr"],
        },
        "handler": calculator,
    },
]
