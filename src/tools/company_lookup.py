"""公司名 → A 股代码。静态映射，防幻觉；查行情/指标前先走这里。"""

from __future__ import annotations

from src.common.types import ToolResult, fail, ok

_YEARS = ["2021", "2022", "2023", "2024", "2025"]

COMPANIES: list[dict[str, object]] = [
    {"name": "贵州茅台", "stock_code": "600519", "years": list(_YEARS)},
    {"name": "五粮液", "stock_code": "000858", "years": list(_YEARS)},
    {"name": "宁德时代", "stock_code": "300750", "years": list(_YEARS)},
    {"name": "海康威视", "stock_code": "002415", "years": list(_YEARS)},
    {"name": "中国平安", "stock_code": "601318", "years": list(_YEARS)},
]

CODE_TO_NAME: dict[str, str] = {str(c["stock_code"]): str(c["name"]) for c in COMPANIES}

_NAME_TO_CODE: dict[str, str] = {
    "贵州茅台": "600519",
    "茅台": "600519",
    "五粮液": "000858",
    "宁德时代": "300750",
    "海康威视": "002415",
    "海康": "002415",
    "中国平安": "601318",
    "平安": "601318",
}

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "company_lookup",
        "description": (
            "将公司中文名转换为 A 股 6 位股票代码。"
            "调用 financial_indicator 或 stock_price 之前必须先用本工具获取代码。"
            "知识库收录：贵州茅台/五粮液/宁德时代/海康威视/中国平安。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "公司中文名或常用简称，如 '贵州茅台'、'茅台'",
                },
            },
            "required": ["name"],
        },
    },
}


def company_lookup(name: str) -> ToolResult:
    """将公司中文名转换为股票代码；精确优先，其次包含匹配。"""
    q = (name or "").strip()
    if not q:
        return fail("name 不能为空")

    if q in CODE_TO_NAME:
        code = q
        official = CODE_TO_NAME[code]
        return ok(
            {
                "query": q,
                "name": official,
                "stock_code": code,
                "match": "code",
            }
        )

    code = _NAME_TO_CODE.get(q)
    if code:
        return ok(
            {
                "query": q,
                "name": CODE_TO_NAME[code],
                "stock_code": code,
                "match": "exact",
            }
        )

    candidates = []
    for alias, c in _NAME_TO_CODE.items():
        if q in alias or alias in q:
            official = CODE_TO_NAME[c]
            item = {"name": official, "stock_code": c, "alias": alias}
            if item not in candidates:
                candidates.append(item)
    if candidates:
        return ok(
            {
                "query": q,
                "match": "fuzzy",
                "candidates": candidates,
                "name": candidates[0]["name"],
                "stock_code": candidates[0]["stock_code"],
            }
        )

    supported = "、".join(f"{c['name']}({c['stock_code']})" for c in COMPANIES)
    return fail(f"未找到 '{q}'，当前支持：{supported}")


def tool_result_text(result: ToolResult) -> str:
    if not result.get("ok"):
        return f"[错误] {result.get('error') or '未知错误'}"
    data = result.get("data") or {}
    if data.get("match") == "fuzzy":
        cands = data.get("candidates") or []
        shown = "、".join(f"{c['name']}({c['stock_code']})" for c in cands)
        return (
            f"未精确匹配 '{data.get('query')}'，按最相近取 "
            f"{data.get('name')} 代码 {data.get('stock_code')}；候选：{shown}"
        )
    return f"{data.get('name')} 的股票代码为 {data.get('stock_code')}"
