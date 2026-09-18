"""
run_fc.py — 方式一：OpenAI 兼容 Function Call

手写 JSON Schema + 进程内 dispatch → src.tools。
单轮闭环：create(tools) → tool_calls → 执行 → role=tool 回填 → 最终回答。

用法：
  python src\\adapters\\function_call\\run_fc.py "茅台2023营收"
  python src\\adapters\\function_call\\run_fc.py --q "北京天气" --provider dashscope
  python src\\adapters\\function_call\\run_fc.py --json --quiet -q "..."
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.adapters.tool_text import dispatch_tool  # noqa: E402
from src.common.config import CHAT_PROVIDERS, get_chat_client  # noqa: E402
from src.tools.rag_backend import (  # noqa: E402
    list_companies,
    rag_search,
    tool_result_text as rag_text,
)
from src.tools.weather_backend import (  # noqa: E402
    query_weather,
    tool_result_text as weather_text,
)

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": (
                "在A股年报语料库中检索与问题最相关的段落。"
                "知识库仅收录：贵州茅台(600519)/五粮液(000858)/宁德时代(300750)/"
                "海康威视(002415)/中国平安(601318)，年份 2021–2025。"
                "不在库内的公司请勿调用本工具。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "检索问题。重要：不要包含公司名和年份"
                            "（已由 stock_code/year 过滤），只用简短财务术语，"
                            "例如 '营收和净利润'、'研发投入'。"
                        ),
                    },
                    "stock_code": {
                        "type": "string",
                        "description": "可选，按公司过滤，如 '600519'",
                    },
                    "year": {
                        "type": "string",
                        "description": "可选，按年份过滤，如 '2023'",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回段落数，默认5",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_companies",
            "description": "列出年报知识库中收录的所有公司、股票代码与可查年份。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_weather",
            "description": "查询指定城市当前天气及未来3天预报。城市用中文名，如 '北京'、'宁德'。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市中文名"},
                },
                "required": ["city"],
            },
        },
    },
]

TOOL_DISPATCH = {
    "rag_search": rag_search,
    "list_companies": list_companies,
    "query_weather": query_weather,
}

TEXT_FNS = {
    "rag_search": rag_text,
    "list_companies": rag_text,
    "query_weather": weather_text,
}

SYSTEM_PROMPT = (
    "你是一名金融分析助手。回答A股年报问题时必须先调用 rag_search 检索原文，"
    "只依据工具返回作答，不要编造。知识库仅含贵州茅台/五粮液/宁德时代/海康威视/中国平安；"
    "不在库内请明确告知。涉及天气时调用 query_weather。本回合可一次调用多个工具。"
)


def run(client, model: str, question: str, verbose: bool = True) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    t0 = time.time()
    tool_call_log: list[dict] = []

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=TOOLS_SCHEMA,
        tool_choice="auto",
    )
    msg = resp.choices[0].message

    if msg.tool_calls:
        messages.append(msg)
        for tc in msg.tool_calls:
            name = tc.function.name
            raw_args = tc.function.arguments or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                args = {}
                tool_call_log.append({"name": name, "args": {"_parse_error": str(e)}})
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": f"参数不是合法 JSON：{e}",
                    }
                )
                continue
            tool_call_log.append({"name": name, "args": args})
            if verbose:
                print(f"  → [tool] {name}({args})")
            result = dispatch_tool(name, args, TOOL_DISPATCH, TEXT_FNS)
            preview = (result or "")[:120].replace("\n", " ")
            if verbose:
                print(f"    ↩ {preview}{'...' if len(result or '') > 120 else ''}\n")
            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": result}
            )

        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
        )
        msg = resp.choices[0].message

    answer = msg.content or ""
    elapsed = time.time() - t0
    if verbose:
        print(f"  → [llm] 最终回答（{elapsed:.1f}s）")
    return {"answer": answer, "tool_calls": tool_call_log, "elapsed": elapsed}


DEMO_QUESTIONS = [
    "贵州茅台2024年营业收入是多少？",
    "宁德时代2024年营收和净利润是多少？另外总部宁德的天气如何？",
    "对比贵州茅台和五粮液2024年的营收。",
    "比亚迪2024年营收是多少？",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="V4 Function Call 适配器")
    parser.add_argument("positional_q", nargs="?", help="问题（位置参数）")
    parser.add_argument("--question", "-q", help="问题")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument(
        "--provider",
        default="dashscope",
        choices=list(CHAT_PROVIDERS.keys()),
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    client, model = get_chat_client(args.provider)
    if not args.json:
        print(f"[Function Call] provider={args.provider} model={model}\n")

    q = args.question or args.positional_q
    questions = DEMO_QUESTIONS if args.demo else ([q] if q else [DEMO_QUESTIONS[0]])
    results = []
    for i, question in enumerate(questions, 1):
        if not args.json:
            print("=" * 60)
            print(f"Q{i}：{question}")
            print("=" * 60)
        result = run(client, model, question, verbose=not (args.quiet or args.json))
        result["question"] = question
        results.append(result)
        if not args.json:
            print("\n最终回答：")
            print(result["answer"])
            print()

    if args.json:
        print(json.dumps(results[0] if len(results) == 1 else results, ensure_ascii=False))


if __name__ == "__main__":
    main()
