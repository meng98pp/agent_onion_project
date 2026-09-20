"""手写 Prompt 解析版 ReAct：Thought → Action → Observation。"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Generator

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import src.common.config  # noqa: F401,E402
from src.common.config import MAX_STEPS, REPEAT_ACTION_LIMIT, get_chat_client  # noqa: E402
from src.tools import execute_tool  # noqa: E402

SYSTEM_PROMPT = """你是一名专业的 A 股金融分析助手，可以使用以下工具：

1. company_lookup(name) - 公司中文名 → 6 位股票代码
2. rag_search(query, stock_code?, year?, top_k?) - 年报语义检索（战略/风险/管理层讨论等定性内容）
3. financial_indicator(symbol) - 近 3 年结构化财务指标（营收/毛利率/ROE 等）
4. stock_price(symbol, start_date, end_date) - 历史股价，日期格式 YYYYMMDD
5. calculator(expr) - 四则运算与幂运算（差值、增长率必须用它，禁止心算）

你必须严格按照以下格式交替输出，每次只能调用一个工具：

Thought: 分析当前状态，决定下一步做什么
Action: 工具名称
Action Input: {"参数名": "参数值"}

收到工具结果后继续推理，直到可以给出最终答案：

Thought: 已有足够信息
Final Answer: 完整的回答（含数据来源）

规则：
- 调用 financial_indicator 或 stock_price 之前，必须先用 company_lookup 获取股票代码
- 数字计算必须用 calculator，不能心算
- rag_search 的 query 用短财务术语，公司与年份尽量走 stock_code/year
- Final Answer 必须引用具体数据来源（年报页码或 AkShare）
- 知识库仅含贵州茅台/五粮液/宁德时代/海康威视/中国平安（2021–2025）
- 没有合适工具能回答时，直接 Final Answer 说明原因，不要编造
"""

_THOUGHT_RE = re.compile(r"Thought:\s*(.+?)(?=\nAction:|\nFinal Answer:|$)", re.DOTALL)
_ACTION_RE = re.compile(r"Action:\s*(\w+)")
_ACTION_INPUT_RE = re.compile(r"Action Input:\s*(\{.+?\})", re.DOTALL)
_FINAL_RE = re.compile(r"Final Answer:\s*(.+)", re.DOTALL)


def _parse_step(text: str) -> dict:
    final = _FINAL_RE.search(text)
    if final:
        thought_m = _THOUGHT_RE.search(text)
        return {
            "type": "final",
            "thought": thought_m.group(1).strip() if thought_m else "",
            "answer": final.group(1).strip(),
        }

    thought_m = _THOUGHT_RE.search(text)
    action_m = _ACTION_RE.search(text)
    input_m = _ACTION_INPUT_RE.search(text)
    if not action_m:
        return {"type": "unparseable", "raw": text}

    try:
        action_input = json.loads(input_m.group(1)) if input_m else {}
        if not isinstance(action_input, dict):
            action_input = {}
    except json.JSONDecodeError:
        action_input = {}

    return {
        "type": "action",
        "thought": thought_m.group(1).strip() if thought_m else "",
        "action": action_m.group(1).strip(),
        "action_input": action_input,
    }


def _action_sig(name: str, args: dict) -> str:
    return f"{name}:{json.dumps(args, ensure_ascii=False, sort_keys=True)}"


def run(
    question: str,
    max_steps: int | None = None,
    provider: str = "dashscope",
) -> Generator[dict, None, None]:
    """ReAct 循环，yield 每步 dict（action / final / error / max_steps）。"""
    client, model = get_chat_client(provider)
    steps = max_steps if max_steps is not None else MAX_STEPS
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    last_sig = ""
    consecutive = 0

    for step in range(1, steps + 1):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            stop=["Observation:"],
        )
        llm_output = (response.choices[0].message.content or "").strip()
        parsed = _parse_step(llm_output)

        if parsed["type"] == "final":
            yield {
                "step": step,
                "type": "final",
                "thought": parsed["thought"],
                "answer": parsed["answer"],
            }
            return

        if parsed["type"] == "unparseable":
            yield {
                "step": step,
                "type": "error",
                "observation": f"格式解析失败，原始输出：{llm_output[:200]}",
            }
            return

        tool_name = parsed["action"]
        tool_args = parsed["action_input"]
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
                "thought": parsed["thought"],
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
            "thought": parsed["thought"],
            "action": tool_name,
            "action_input": tool_args,
            "observation": observation,
        }
        messages.append({"role": "assistant", "content": llm_output})
        messages.append({"role": "user", "content": f"Observation: {observation}\n"})

    yield {
        "step": steps + 1,
        "type": "max_steps",
        "answer": f"已达最大步数 {steps}，未能得出最终答案",
    }
