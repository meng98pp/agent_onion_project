"""AkShare A 股区间行情（前复权）。日期格式 YYYYMMDD。"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from typing import Any

from src.common.config import AKSHARE_TIMEOUT_SEC
from src.common.types import ToolResult, fail, ok

_DATE_RE = re.compile(r"^\d{8}$")

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "stock_price",
        "description": (
            "获取 A 股历史股价及区间涨跌幅（前复权）。"
            "日期格式必须是 YYYYMMDD，例如 20230101。"
            "symbol 必须是 6 位股票代码，请先调用 company_lookup。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "6 位股票代码，如 '600519'",
                },
                "start_date": {
                    "type": "string",
                    "description": "起始日期 YYYYMMDD，如 '20230101'",
                },
                "end_date": {
                    "type": "string",
                    "description": "结束日期 YYYYMMDD，如 '20231231'",
                },
            },
            "required": ["symbol", "start_date", "end_date"],
        },
    },
}


def _fetch(symbol: str, start_date: str, end_date: str) -> Any:
    import akshare as ak

    return ak.stock_zh_a_hist(
        symbol=symbol,
        period="daily",
        start_date=start_date,
        end_date=end_date,
        adjust="qfq",
    )


def _run_akshare(symbol: str, start_date: str, end_date: str) -> Any:
    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(_fetch, symbol, start_date, end_date)
        try:
            return fut.result(timeout=AKSHARE_TIMEOUT_SEC)
        except FuturesTimeout:
            raise TimeoutError(f"AkShare 超时（{AKSHARE_TIMEOUT_SEC:.0f}s）") from None


def stock_price(symbol: str, start_date: str, end_date: str) -> ToolResult:
    code = (symbol or "").strip()
    start = (start_date or "").strip()
    end = (end_date or "").strip()
    if not code:
        return fail("symbol 不能为空")
    if not (code.isdigit() and len(code) == 6):
        return fail("symbol 须为 6 位股票代码，请先 company_lookup")
    if not _DATE_RE.match(start) or not _DATE_RE.match(end):
        return fail("日期格式必须是 YYYYMMDD，例如 20230101")
    if start > end:
        return fail("start_date 不能晚于 end_date")

    try:
        df = _run_akshare(code, start, end)
    except Exception as e:
        return fail(f"stock_price 执行出错: {e}")

    if df is None or getattr(df, "empty", True):
        return fail(f"未获取到 {code} 在 {start}~{end} 的行情数据")

    try:
        first_close = float(df.iloc[0]["收盘"])
        last_close = float(df.iloc[-1]["收盘"])
        high = float(df["最高"].max())
        low = float(df["最低"].min())
    except Exception as e:
        return fail(f"行情字段解析失败: {e}")

    if first_close == 0:
        return fail("区间起始收盘价为 0，无法计算涨跌幅")
    change_pct = (last_close - first_close) / first_close * 100

    return ok(
        {
            "symbol": code,
            "start_date": start,
            "end_date": end,
            "first_close": round(first_close, 2),
            "last_close": round(last_close, 2),
            "high": round(high, 2),
            "low": round(low, 2),
            "change_pct": round(change_pct, 2),
        }
    )


def tool_result_text(result: ToolResult) -> str:
    if not result.get("ok"):
        return f"[错误] {result.get('error') or '未知错误'}"
    d = result.get("data") or {}
    return (
        f"股票代码: {d.get('symbol')}，区间: {d.get('start_date')}~{d.get('end_date')}\n"
        f"  区间起始收盘价: {d.get('first_close'):.2f} 元\n"
        f"  区间末尾收盘价: {d.get('last_close'):.2f} 元\n"
        f"  区间最高价: {d.get('high'):.2f} 元\n"
        f"  区间最低价: {d.get('low'):.2f} 元\n"
        f"  区间涨跌幅: {d.get('change_pct'):+.2f}%"
    )
