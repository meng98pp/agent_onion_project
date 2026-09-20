"""AkShare 近三年关键财务指标。失败返回 ToolResult，不抛到 Agent。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from typing import Any

from src.common.config import AKSHARE_TIMEOUT_SEC
from src.common.types import ToolResult, fail, ok

_TARGET_ROWS = (
    "归母净利润",
    "营业总收入",
    "毛利率",
    "净利率",
    "净资产收益率",
    "资产负债率",
    "每股收益",
)

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "financial_indicator",
        "description": (
            "获取 A 股近 3 年关键财务指标（营收/净利润/毛利率/ROE/资产负债率等），"
            "适合跨年对比或与年报 RAG 交叉验证。symbol 必须是 6 位股票代码，"
            "请先调用 company_lookup。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "6 位股票代码，如 '600519'",
                },
            },
            "required": ["symbol"],
        },
    },
}


def _fetch(symbol: str) -> Any:
    import akshare as ak

    return ak.stock_financial_abstract(symbol=symbol)


def _run_akshare(symbol: str) -> Any:
    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(_fetch, symbol)
        try:
            return fut.result(timeout=AKSHARE_TIMEOUT_SEC)
        except FuturesTimeout:
            raise TimeoutError(f"AkShare 超时（{AKSHARE_TIMEOUT_SEC:.0f}s）") from None


def financial_indicator(symbol: str) -> ToolResult:
    code = (symbol or "").strip()
    if not code:
        return fail("symbol 不能为空")
    if not (code.isdigit() and len(code) == 6):
        return fail("symbol 须为 6 位股票代码，请先 company_lookup")

    try:
        df = _run_akshare(code)
    except Exception as e:
        return fail(f"financial_indicator 执行出错: {e}")

    if df is None or getattr(df, "empty", True):
        return fail(f"未获取到 {code} 的财务指标数据")

    date_cols = [c for c in df.columns if str(c).endswith("1231")][:3]
    if not date_cols:
        date_cols = [c for c in df.columns[2:5]]

    indicators: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        label = str(row.get("指标", ""))
        if not any(t in label for t in _TARGET_ROWS):
            continue
        values: dict[str, str] = {}
        for col in date_cols:
            raw = row.get(col)
            try:
                values[str(col)[:4]] = f"{float(raw):.4g}"
            except (TypeError, ValueError):
                values[str(col)[:4]] = str(raw)
        indicators.append({"name": label, "values": values})

    if not indicators:
        return fail(f"{code} 未找到关键财务指标行")

    return ok(
        {
            "symbol": code,
            "periods": [str(c)[:4] for c in date_cols],
            "indicators": indicators,
        }
    )


def tool_result_text(result: ToolResult) -> str:
    if not result.get("ok"):
        return f"[错误] {result.get('error') or '未知错误'}"
    data = result.get("data") or {}
    lines = [f"股票代码: {data.get('symbol')}，数据截至最近三个年报"]
    for item in data.get("indicators") or []:
        vals = " | ".join(f"{y}年: {v}" for y, v in (item.get("values") or {}).items())
        lines.append(f"  {item.get('name')}: {vals}")
    return "\n".join(lines)
