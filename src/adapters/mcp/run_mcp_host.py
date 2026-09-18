"""
run_mcp_host.py — 方式二：MCP Host

连接 rag/weather Server（stdio）→ list_tools 发现 → call_tool 跨进程执行。
天气场景支持多轮链式（geocode → fetch）。

用法：
  python src\\adapters\\mcp\\run_mcp_host.py --q "茅台2024营收"
  python src\\adapters\\mcp\\run_mcp_host.py --demo
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from contextlib import AsyncExitStack
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.common.config import CHAT_PROVIDERS, get_chat_client  # noqa: E402

SERVERS_DIR = Path(__file__).resolve().parent / "servers"
MAX_TOOL_ROUNDS = 6

SYSTEM_PROMPT = (
    "你是一名金融分析助手。回答A股年报问题时必须先调用 rag_search 检索原文，"
    "只依据工具返回作答，不要编造。知识库仅含贵州茅台/五粮液/宁德时代/海康威视/中国平安；"
    "不在库内请明确告知。"
    "涉及天气时必须两步："
    "① geocode_city(city) 取 latitude/longitude；"
    "② fetch_weather(latitude, longitude)。"
    "不要跳过地理编码。工具结果足够后直接文字回答。"
)


def build_server_configs() -> dict[str, StdioServerParameters]:
    return {
        "rag": StdioServerParameters(
            command=sys.executable,
            args=[str(SERVERS_DIR / "rag_server.py")],
            env={**os.environ},
        ),
        "weather": StdioServerParameters(
            command=sys.executable,
            args=[str(SERVERS_DIR / "weather_server.py")],
            env={**os.environ},
        ),
    }


async def connect_all_servers(stack: AsyncExitStack):
    print("正在连接 MCP Servers...\n", file=sys.stderr)
    tool_registry: dict[str, tuple[ClientSession, str]] = {}
    openai_tools: list[dict] = []

    for label, params in build_server_configs().items():
        read, write = await stack.enter_async_context(stdio_client(params))
        session: ClientSession = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        tools_result = await session.list_tools()
        for tool in tools_result.tools:
            tool_registry[tool.name] = (session, label)
            # mcp 2.x 把 v1 的 inputSchema 改成了 Python 属性 input_schema
            schema = getattr(tool, "input_schema", None)
            if schema is None:
                schema = getattr(tool, "inputSchema", None)
            if hasattr(schema, "model_dump"):
                schema = schema.model_dump(exclude_none=True)
            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": schema or {"type": "object", "properties": {}},
                    },
                }
            )
        names = ", ".join(t.name for t in tools_result.tools)
        print(f"  ✓ [{label}]  {names}", file=sys.stderr)

    print(f"\n共 {len(tool_registry)} 个工具就绪\n", file=sys.stderr)
    return tool_registry, openai_tools


async def run(
    client,
    model: str,
    question: str,
    tool_registry: dict,
    openai_tools: list[dict],
    verbose: bool = True,
) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    t0 = time.time()
    tool_call_log: list[dict] = []

    for round_idx in range(1, MAX_TOOL_ROUNDS + 1):
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=openai_tools,
            tool_choice="auto",
        )
        msg = resp.choices[0].message

        if not msg.tool_calls:
            answer = msg.content or ""
            elapsed = time.time() - t0
            if verbose:
                print(f"  → [llm] 最终回答（共 {round_idx} 轮请求，{elapsed:.1f}s）")
            return {"answer": answer, "tool_calls": tool_call_log, "elapsed": elapsed}

        if verbose:
            print(f"  —— 工具轮次 {round_idx} ——")
        messages.append(msg)
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            tool_call_log.append({"name": name, "args": args, "round": round_idx})
            if verbose:
                print(f"  → [mcp] {name}({args})")

            session, label = tool_registry.get(name, (None, None))
            if session is None:
                result = f"未知工具：{name}"
            else:
                call_result = await session.call_tool(name, args)
                result = "\n".join(
                    b.text for b in call_result.content if hasattr(b, "text")
                )

            preview = (result or "")[:120].replace("\n", " ")
            if verbose:
                print(f"    ↩ [{label}] {preview}{'...' if len(result or '') > 120 else ''}\n")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    if verbose:
        print(f"  ⚠ 已达 MAX_TOOL_ROUNDS={MAX_TOOL_ROUNDS}，强制生成最终回答", file=sys.stderr)
    resp = client.chat.completions.create(model=model, messages=messages)
    answer = resp.choices[0].message.content or ""
    elapsed = time.time() - t0
    if verbose:
        print(f"  → [llm] 最终回答（达轮次上限，{elapsed:.1f}s）")
    return {"answer": answer, "tool_calls": tool_call_log, "elapsed": elapsed}


DEMO_QUESTIONS = [
    "贵州茅台2024年营业收入是多少？",
    "宁德时代2024年营收和净利润是多少？另外总部宁德的天气如何？",
    "对比贵州茅台和五粮液2024年的营收。",
    "比亚迪2024年营收是多少？",
]


async def main_async(
    provider: str,
    question: str | None,
    demo: bool,
    verbose: bool,
    as_json: bool,
) -> None:
    client, model = get_chat_client(provider)
    if not as_json:
        print(f"[MCP] provider={provider} model={model}\n", file=sys.stderr)

    async with AsyncExitStack() as stack:
        tool_registry, openai_tools = await connect_all_servers(stack)
        questions = (
            DEMO_QUESTIONS if demo else ([question] if question else [DEMO_QUESTIONS[0]])
        )
        results = []
        for i, q in enumerate(questions, 1):
            if not as_json:
                print("=" * 60)
                print(f"Q{i}：{q}")
                print("=" * 60)
            result = await run(
                client,
                model,
                q,
                tool_registry,
                openai_tools,
                verbose=verbose and not as_json,
            )
            result["question"] = q
            results.append(result)
            if not as_json:
                print("\n最终回答：")
                print(result["answer"])
                print()

        if as_json:
            print(
                json.dumps(
                    results[0] if len(results) == 1 else results,
                    ensure_ascii=False,
                )
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="V4 MCP Host")
    parser.add_argument("positional_q", nargs="?", help="问题")
    parser.add_argument("--question", "-q")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument(
        "--provider",
        default="dashscope",
        choices=list(CHAT_PROVIDERS.keys()),
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    q = args.question or args.positional_q
    asyncio.run(
        main_async(
            args.provider,
            q,
            args.demo,
            verbose=not args.quiet,
            as_json=args.json,
        )
    )


if __name__ == "__main__":
    main()
