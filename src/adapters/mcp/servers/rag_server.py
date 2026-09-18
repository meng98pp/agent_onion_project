"""
rag_server.py — 年报 RAG MCP Server（stdio）

只做 @mcp.tool 接线；检索算法在 src.tools.rag_backend。
日志必须写 stderr（stdout 是 JSON-RPC 通道）。
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mcp.server.mcpserver import MCPServer  # noqa: E402

from src.tools.rag_backend import (  # noqa: E402
    list_companies as _list_companies,
    rag_search as _rag_search,
    tool_result_text as _rag_text,
)


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


mcp = MCPServer("rag-server")


@mcp.tool()
def rag_search(
    query: str,
    stock_code: str | None = None,
    year: str | None = None,
    top_k: int = 5,
) -> str:
    """
    在A股年报语料库中检索与问题最相关的段落。

    知识库仅收录：贵州茅台(600519)/五粮液(000858)/宁德时代(300750)/
    海康威视(002415)/中国平安(601318)，年份 2021–2025。不在库内请勿调用。

    Args:
        query: 检索问题。不要含公司名/年份，用短财务术语如 '营收和净利润'。
        stock_code: 可选股票代码过滤。
        year: 可选年份过滤。
        top_k: 返回条数，默认5。
    """
    return _rag_text(_rag_search(query, top_k, stock_code, year))


@mcp.tool()
def list_companies() -> str:
    """列出年报知识库收录的公司、股票代码与可查年份。"""
    return _rag_text(_list_companies())


if __name__ == "__main__":
    log("RAG MCP Server 启动中（stdio）...")
    mcp.run(transport="stdio")
