"""原生 Function Calling 版 ReAct；工具 Schema 按已解锁 Skill 渐进披露。"""

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
from src.harness.tool_registry import (  # noqa: E402
    call_tool,
    get_tools_schema,
    is_load_skill_success,
)

SYSTEM_PROMPT = """你是一名专业的 A 股金融分析助手，运行在 Harness 约束内。
规则：
- System Prompt 里只有技能目录（L0）。必须先 load_skill(name)，才能调用该技能的业务工具
- 可用元工具：list_skills / load_skill / read_skill_resource
- 用户问题匹配 L0 某技能时：立刻调用 load_skill，成功后再调业务工具。禁止询问「是否启用」，禁止只口头描述工具而不调用
- 问天气/气温/下雨 → load_skill("weather-query") → query_weather(city)
- 问财务/股价/年报 → load_skill("stock-analyst") → 再按技能说明书调用
- 调用 financial_indicator 或 stock_price 之前，必须先用 company_lookup 获取股票代码
- 数字计算必须使用 calculator 工具，不能心算
- rag_search 的 query 用短财务术语，公司与年份尽量走 stock_code/year
- 最终回答必须引用具体数据来源（年报页码、AkShare 或天气工具返回）
- 知识库仅含贵州茅台/五粮液/宁德时代/海康威视/中国平安（2021–2025）
- 没有合适工具能回答时，直接说明原因，不要编造
- 禁止路径穿越、禁止调用未解锁的业务工具
"""


def _action_sig(name: str, args: dict) -> str:
    return f"{name}:{json.dumps(args, ensure_ascii=False, sort_keys=True)}"


def run(
    question: str,
    max_steps: int | None = None,
    provider: str = "dashscope",
    extra_system: str | None = None,
    history: list[dict] | None = None,
) -> Generator[dict, None, None]:
    """与 react_manual.run() 同形的 step dict，便于 evaluate / serve 对照。"""
    from src.agent.loop import compose_system_prompt

    client, model = get_chat_client(provider)
    steps = max_steps if max_steps is not None else MAX_STEPS
    system = compose_system_prompt(SYSTEM_PROMPT, extra_system)
    messages: list[dict] = [{"role": "system", "content": system}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": question})
    last_sig = ""
    consecutive = 0
    activated: set[str] = set()

    for step in range(1, steps + 1):
        schema = get_tools_schema(activated_skills=activated)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=schema,
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
                "activated_skills": sorted(activated),
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
                    "activated_skills": sorted(activated),
                }
                return

            observation = call_tool(tool_name, tool_args, activated_skills=activated)
            if tool_name == "load_skill":
                skill_name = str(tool_args.get("name") or "").strip()
                if skill_name and is_load_skill_success(observation, skill_name):
                    activated.add(skill_name)
                    observation = (
                        observation
                        + f"\n\n[system] 已解锁技能 `{skill_name}` 的业务工具；"
                        "请在后续步骤中直接调用它们。"
                    )

            yield {
                "step": step,
                "type": "action",
                "thought": "",
                "action": tool_name,
                "action_input": tool_args,
                "observation": observation,
                "activated_skills": sorted(activated),
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
        "activated_skills": sorted(activated),
    }
