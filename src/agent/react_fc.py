"""原生 Function Calling 版 ReAct；Thought 在模型内部不可见。"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Generator

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import src.common.config  # noqa: F401,E402
from src.common.config import MAX_STEPS, REPEAT_ACTION_LIMIT, get_chat_client  # noqa: E402
from src.tools import TOOLS_SCHEMA, execute_tool  # noqa: E402

SYSTEM_PROMPT = """你是一名专业的 A 股金融分析助手。
规则：
- 调用 financial_indicator 或 stock_price 之前，必须先用 company_lookup 获取股票代码
- 数字计算必须使用 calculator 工具，不能心算
- rag_search 的 query 用短财务术语，公司与年份尽量走 stock_code/year
- 最终回答必须引用具体数据来源（年报页码或 AkShare）
- 知识库仅含贵州茅台/五粮液/宁德时代/海康威视/中国平安（2021–2025）
- 没有合适工具能回答时，直接说明原因，不要编造
"""


def _action_sig(name: str, args: dict) -> str:
    return f"{name}:{json.dumps(args, ensure_ascii=False, sort_keys=True)}"


def run(
    question: str,
    max_steps: int | None = None,
    provider: str = "dashscope",
    extra_system: str | None = None,
) -> Generator[dict, None, None]:
    """与 react_manual.run() 同形的 step dict，便于 evaluate / serve 对照。"""
    client, model = get_chat_client(provider)
    steps = max_steps if max_steps is not None else MAX_STEPS
    system = SYSTEM_PROMPT
    if extra_system:
        system = SYSTEM_PROMPT + "\n\n---\n\n" + extra_system
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    last_sig = ""
    consecutive = 0

    for step in range(1, steps + 1):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
            temperature=0,
        )
        msg = response.choices[0].message
        reason = response.choices[0].finish_reason

        if reason == "stop" or not msg.tool_calls:
            yield {
                "step": step,
                "type": "final",
                "thought": "",
                "answer": msg.content or "（模型返回空内容）",
            }
            return

        messages.append(msg)

        for tool_call in msg.tool_calls:
            tool_name = tool_call.function.name
            raw_args = tool_call.function.arguments or "{}"
            try:
                tool_args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
                if not isinstance(tool_args, dict):
                    tool_args = {}
            except (json.JSONDecodeError, TypeError, ValueError):
                tool_args = {}

            sig = _action_sig(tool_name, tool_args)
            if sig == last_sig:
                consecutive += 1
            else:
                consecutive = 1
                last_sig = sig
            if consecutive >= REPEAT_ACTION_LIMIT:
                yield {
                    "step": step,
                    "type": "error",
                    "thought": "",
                    "action": tool_name,
                    "action_input": tool_args,
                    "observation": (
                        f"连续相同 Action 熔断：{tool_name}({json.dumps(tool_args, ensure_ascii=False)})"
                    ),
                }
                return

            _result, observation = execute_tool(tool_name, tool_args)
            yield {
                "step": step,
                "type": "action",
                "thought": "",
                "action": tool_name,
                "action_input": tool_args,
                "observation": observation,
            }
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": observation,
                }
            )

    yield {
        "step": steps + 1,
        "type": "max_steps",
        "answer": f"已达最大步数 {steps}，未能得出最终答案",
    }
