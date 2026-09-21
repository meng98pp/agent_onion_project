"""手写 Prompt 解析版 ReAct：Thought → Action → Observation；Skill 渐进披露。"""

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
from src.harness.tool_registry import call_tool, is_load_skill_success  # noqa: E402

SYSTEM_PROMPT = """你是一名专业的 A 股金融分析助手，运行在 Harness 约束内。

初始只有元工具（业务工具必须先 load_skill 才能用）：
1. list_skills() - 列出已发现技能（L0）
2. load_skill(name) - 加载技能说明书（L1）并解锁该技能 tools.py 中的业务工具
3. read_skill_resource(skill, path) - 读取技能目录内参考文件（L2），禁止 .. 穿越

你必须严格按照以下格式交替输出，每次只能调用一个工具：

Thought: 分析当前状态，决定下一步做什么
Action: 工具名称
Action Input: {"参数名": "参数值"}

收到工具结果后继续推理，直到可以给出最终答案：

Thought: 已有足够信息
Final Answer: 完整的回答（含数据来源）

规则：
- 必须先 load_skill，再调用该技能业务工具（未解锁会被拒绝）
- 用户问题匹配 L0 某技能时：立刻 Action: load_skill，禁止询问「是否启用」，禁止只口头描述
- 问天气 → load_skill({"name":"weather-query"}) → query_weather({"city":"北京"})
- 问财务/股价/年报 → load_skill({"name":"stock-analyst"})
- 调用 financial_indicator 或 stock_price 之前，必须先用 company_lookup 获取股票代码
- 数字计算必须用 calculator，不能心算
- rag_search 的 query 用短财务术语，公司与年份尽量走 stock_code/year
- Final Answer 必须引用具体数据来源（年报页码、AkShare 或天气工具返回）
- 知识库仅含贵州茅台/五粮液/宁德时代/海康威视/中国平安（2021–2025）
- 没有合适工具能回答时，直接 Final Answer 说明原因，不要编造
"""

_THOUGHT_RE = re.compile(r"Thought:\s*(.+?)(?=\nAction:|\nFinal Answer:|$)", re.DOTALL)
_ACTION_RE = re.compile(r"Action:\s*([A-Za-z_][A-Za-z0-9_]*)")
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
    extra_system: str | None = None,
    history: list[dict] | None = None,
) -> Generator[dict, None, None]:
    """ReAct 循环，yield 每步 dict（action / final / error / max_steps）。"""
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
                "activated_skills": sorted(activated),
            }
            return

        if parsed["type"] == "unparseable":
            yield {
                "step": step,
                "type": "error",
                "observation": f"格式解析失败，原始输出：{llm_output[:200]}",
                "activated_skills": sorted(activated),
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
            "thought": parsed["thought"],
            "action": tool_name,
            "action_input": tool_args,
            "observation": observation,
            "activated_skills": sorted(activated),
        }
        messages.append({"role": "assistant", "content": llm_output})
        messages.append({"role": "user", "content": f"Observation: {observation}\n"})

    yield {
        "step": steps + 1,
        "type": "max_steps",
        "answer": f"已达最大步数 {steps}，未能得出最终答案",
        "activated_skills": sorted(activated),
    }
