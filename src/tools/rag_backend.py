"""
rag_backend — A股年报检索业务后端（FC / MCP / CLI 共享）

只做检索与格式化，不感知协议。复用 V3 `src.rag.retriever.search`，禁止在适配器里重写算法。
对外统一返回 ToolResult：{ok, data, error}。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import src.common.config  # noqa: F401,E402 — 加载 .env + OpenMP 兼容开关
from src.common.types import ToolResult, fail, ok  # noqa: E402
from src.rag.retriever import search as _vector_search  # noqa: E402
from src.tools.company_lookup import CODE_TO_NAME, COMPANIES  # noqa: E402


def _format_hit(hit: dict[str, Any]) -> dict[str, Any]:
    code = str(hit.get("stock_code") or "")
    return {
        "score": float(hit.get("score") or 0.0),
        "text": hit.get("text") or hit.get("content") or "",
        "stock_code": code,
        "company": CODE_TO_NAME.get(code, code),
        "year": str(hit.get("year") or ""),
        "section": hit.get("section_path") or hit.get("section") or "",
        "page_num": hit.get("page_num"),
        "source": hit.get("source") or "",
        "chunk_id": hit.get("chunk_id") or "",
    }


def _substantive(text: str) -> bool:
    """标题块（如「营业收入和营业成本」）会抢走 Top1，但没有数字，模型只能回答没找到。"""
    t = (text or "").strip()
    return len(t) >= 40 and any(ch.isdigit() for ch in t)


def rag_search(
    query: str,
    top_k: int = 5,
    stock_code: str | None = None,
    year: str | None = None,
) -> ToolResult:
    """
    在年报向量库中检索相关段落。

    Args:
        query: 检索问题；建议不含公司名/年份（由 stock_code/year 过滤），用短财务术语。
        top_k: 返回条数，默认 5。
        stock_code: 可选股票代码过滤。
        year: 可选年份过滤（字符串，如 "2023"）。
    """
    q = (query or "").strip()
    if not q:
        return fail("query 不能为空")
    if top_k <= 0:
        return fail("top_k 必须为正整数")

    try:
        # 标题块常排在数字表前面，多取再挑含数字的段落
        fetch_k = min(max(top_k * 20, 80) if (stock_code or year) else top_k * 8, 200)
        raw_hits = _vector_search(q, k=fetch_k)
    except Exception as e:
        return fail(f"检索失败：{e}")

    useful: list[dict[str, Any]] = []
    fallback: list[dict[str, Any]] = []
    for h in raw_hits:
        if stock_code and str(h.get("stock_code") or "") != str(stock_code):
            continue
        if year and str(h.get("year") or "") != str(year):
            continue
        item = _format_hit(h)
        if _substantive(item["text"]):
            useful.append(item)
        else:
            fallback.append(item)

    hits = (useful + fallback)[:top_k]

    if not hits:
        parts = []
        if stock_code:
            parts.append(f"stock_code={stock_code}")
        if year:
            parts.append(f"year={year}")
        filt = f"（过滤：{', '.join(parts)}）" if parts else ""
        return ok({"hits": [], "count": 0, "message": f"未找到相关内容{filt}"})

    return ok({"hits": hits, "count": len(hits)})


def list_companies() -> ToolResult:
    """列出知识库收录的公司与可查年份。"""
    try:
        return ok({"companies": COMPANIES, "count": len(COMPANIES)})
    except Exception as e:
        return fail(str(e))


def tool_result_text(result: ToolResult) -> str:
    """把 ToolResult 转成 LLM / CLI stdout 可读文本。"""
    if not result.get("ok"):
        return f"[错误] {result.get('error') or '未知错误'}"

    data = result.get("data")
    if not isinstance(data, dict):
        return str(data)

    if "companies" in data:
        lines = ["年报知识库收录公司："]
        for c in data["companies"]:
            years = " / ".join(c.get("years") or [])
            lines.append(f"  {c['name']}  代码：{c['stock_code']}  年份：{years}")
        lines.append(f"共 {data.get('count', len(data['companies']))} 家公司")
        return "\n".join(lines)

    if "hits" in data:
        hits = data.get("hits") or []
        if not hits:
            return str(data.get("message") or "未找到相关内容")
        lines = [f"检索到 {len(hits)} 条相关段落：\n"]
        for i, h in enumerate(hits, 1):
            lines.append(
                f"【{i}】{h.get('company')}（{h.get('stock_code')}）{h.get('year')}年报"
                f" | 第{h.get('page_num')}页 | 相关度：{float(h.get('score') or 0):.3f}"
            )
            if h.get("section"):
                lines.append(f"章节：{h['section']}")
            lines.append(h.get("text") or "")
            lines.append("")
        return "\n".join(lines)

    return str(data)


# 兼容旧名（week11）
search_annual_report = rag_search


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="rag_backend 自检")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("search")
    p.add_argument("--query", required=True)
    p.add_argument("--stock-code", default=None)
    p.add_argument("--year", default=None)
    p.add_argument("--top-k", type=int, default=5)
    sub.add_parser("list-companies")
    args = parser.parse_args()

    if args.cmd == "search":
        r = rag_search(args.query, args.top_k, args.stock_code, args.year)
    else:
        r = list_companies()
    print(json.dumps(r, ensure_ascii=False, indent=2))
    print("---")
    print(tool_result_text(r))
