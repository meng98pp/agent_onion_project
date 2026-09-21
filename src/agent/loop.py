"""V5/V7 统一入口：切换 manual / fc；L0 技能索引注入 system，L1 按需展开。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import src.common.config  # noqa: F401,E402
from src.common.config import CHAT_PROVIDERS, MAX_STEPS  # noqa: E402
from src.harness.skill_loader import load_skill_index  # noqa: E402
from src.harness.tool_registry import build_tools_prompt_section  # noqa: E402

COLORS = {
    "thought": "\033[36m",
    "action": "\033[33m",
    "obs": "\033[32m",
    "final": "\033[35m",
    "error": "\033[31m",
    "reset": "\033[0m",
}

DEFAULT_QUESTION = "用股票分析技能解读宁德时代风险因素"


def _c(color: str, text: str) -> str:
    return f"{COLORS[color]}{text}{COLORS['reset']}"


def compose_system_prompt(base: str, extra_system: str | None = None) -> str:
    """L0 索引进 system；SKILL 正文只在 load_skill 命中后进入 Observation。"""
    parts = [base, build_tools_prompt_section()]
    if extra_system:
        parts.append(extra_system)
    return "\n\n---\n\n".join(part for part in parts if part)


def run_and_print(
    question: str,
    mode: str = "fc",
    max_steps: int | None = None,
    provider: str = "dashscope",
) -> dict:
    if mode == "manual":
        from src.agent.react_manual import run as react_run
    elif mode == "fc":
        from src.agent.react_fc import run as react_run
    else:
        raise ValueError(f"未知 mode: {mode!r}，可选 manual / fc")

    steps_limit = max_steps if max_steps is not None else MAX_STEPS
    skill_names = [row["name"] for row in load_skill_index()]
    print(f"\n{'=' * 60}")
    print(f"问题: {question}")
    print(f"实现: {'手写Prompt解析' if mode == 'manual' else 'Function Calling'}  mode={mode}")
    print(f"Skill L0: {skill_names or '(空)'}")
    print("=" * 60)

    start = time.time()
    final: dict = {"answer": "", "steps": []}

    for step_data in react_run(question, max_steps=steps_limit, provider=provider):
        final["steps"].append(step_data)
        stype = step_data["type"]
        if stype == "action":
            print(f"\n[Step {step_data['step']}]")
            thought = step_data.get("thought") or "（模型内部推理，Function Calling 版不可见）"
            print(_c("thought", f"Thought: {thought}"))
            print(_c("action", f"Action:  {step_data['action']}"))
            print(
                _c(
                    "action",
                    f"Input:   {json.dumps(step_data.get('action_input') or {}, ensure_ascii=False)}",
                )
            )
            obs = str(step_data.get("observation") or "")
            preview = obs if len(obs) <= 400 else obs[:400] + "..."
            print(_c("obs", f"Obs:     {preview}"))
        elif stype == "final":
            elapsed = time.time() - start
            print(f"\n{'-' * 60}")
            if step_data.get("thought"):
                print(_c("thought", f"Thought: {step_data['thought']}"))
            print(_c("final", f"\nFinal Answer:\n{step_data['answer']}"))
            print(f"\n共 {step_data['step']} 步，耗时 {elapsed:.1f}s")
            final["answer"] = step_data["answer"]
            final["elapsed"] = elapsed
        elif stype in ("error", "max_steps"):
            msg = step_data.get("answer") or step_data.get("observation") or ""
            print(_c("error", f"\n{msg}"))
            final["answer"] = msg
            final["elapsed"] = time.time() - start

    return final


def main() -> None:
    parser = argparse.ArgumentParser(description="V7 ReAct + Skill 渐进披露")
    parser.add_argument("--mode", choices=["manual", "fc"], default="fc")
    parser.add_argument("--q", "--question", dest="question", default=DEFAULT_QUESTION)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--provider", default="dashscope", choices=list(CHAT_PROVIDERS.keys()))
    args = parser.parse_args()
    run_and_print(args.question, mode=args.mode, max_steps=args.max_steps, provider=args.provider)


if __name__ == "__main__":
    main()
