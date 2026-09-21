"""
把 tool_registry 导出为 MCP Server（stdio）。

Harness 内对话：Function Calling + load_skill 后才解锁业务 Schema。
MCP 导出：元工具 + 全部已发现业务工具，便于宿主一次性 list/call。
协议与 V4 相同：stdio JSON-RPC。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import src.common.config  # noqa: F401,E402
from src.harness.skill_loader import SkillLoader, bind_skill_loader  # noqa: E402
from src.harness.tool_registry import (  # noqa: E402
    META_TOOL_NAMES,
    call_tool,
    get_tools_schema,
)


def _make_handler(tool_name: str, description: str):
    def _handler(**kwargs: Any) -> str:
        return call_tool(tool_name, kwargs, activated_skills=None)

    _handler.__name__ = tool_name.replace("-", "_")
    _handler.__doc__ = description
    return _handler


def create_mcp_server(name: str = "finagent-harness"):
    bind_skill_loader(SkillLoader())
    schema = get_tools_schema(include_all_skill_tools=True)

    try:
        from mcp.server.fastmcp import FastMCP

        server = FastMCP(
            name,
            instructions=(
                "导出 FinAgent Harness 工具（含 skills/*/tools.py 热插拔发现的业务工具）。"
                "Harness 内对话仍用 Function Calling，且需 load_skill 后才解锁业务 Schema；"
                "本 MCP 进程为便于宿主复用，默认 list 全部已发现工具。"
            ),
        )
        for item in schema:
            fn = item.get("function") or {}
            tool_name = fn.get("name")
            if not tool_name:
                continue
            desc = fn.get("description") or tool_name
            server.add_tool(_make_handler(tool_name, desc), name=tool_name, description=desc)
        return server
    except ImportError:
        from mcp.server.mcpserver import MCPServer

        server = MCPServer(name)
        for item in schema:
            fn = item.get("function") or {}
            tool_name = fn.get("name")
            if not tool_name:
                continue
            desc = fn.get("description") or tool_name
            handler = _make_handler(tool_name, desc)
            decorator = getattr(server, "tool", None)
            if callable(decorator):
                try:
                    decorator(name=tool_name)(handler)
                except TypeError:
                    decorator()(handler)
        return server


def smoke_test() -> None:
    """只断言 Schema / 渐进披露结构，不发起网络请求。"""
    bind_skill_loader(SkillLoader())
    only_meta = {item["function"]["name"] for item in get_tools_schema(activated_skills=set())}
    if only_meta != set(META_TOOL_NAMES):
        raise SystemExit(f"[smoke] 未激活时应仅元工具，实际: {only_meta}")

    stock = {item["function"]["name"] for item in get_tools_schema(activated_skills={"stock-analyst"})}
    if "company_lookup" not in stock or "query_weather" in stock:
        raise SystemExit(f"[smoke] stock-analyst 解锁集合异常: {stock}")

    weather = {
        item["function"]["name"] for item in get_tools_schema(activated_skills={"weather-query"})
    }
    if "query_weather" not in weather:
        raise SystemExit(f"[smoke] weather-query 未解锁 query_weather: {weather}")

    all_names = {
        item["function"]["name"] for item in get_tools_schema(include_all_skill_tools=True)
    }
    expected = set(META_TOOL_NAMES) | {
        "company_lookup",
        "rag_search",
        "financial_indicator",
        "stock_price",
        "calculator",
        "query_weather",
    }
    missing = expected - all_names
    if missing:
        raise SystemExit(f"[smoke] 全量导出缺少工具: {missing}")

    print(f"[smoke] meta={sorted(only_meta)}")
    print(f"[smoke] all ({len(all_names)}): {', '.join(sorted(all_names))}")
    print("[smoke] OK（渐进披露 schema 断言）")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export harness tool_registry as MCP Server")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="进程内 schema 自检后退出（不进入 stdio）",
    )
    args = parser.parse_args(argv)

    if args.smoke:
        smoke_test()
        return

    server = create_mcp_server()
    if hasattr(server, "run"):
        server.run(transport="stdio")
        return
    raise SystemExit("当前 MCP Server 不支持 run(transport='stdio')")


if __name__ == "__main__":
    main()
