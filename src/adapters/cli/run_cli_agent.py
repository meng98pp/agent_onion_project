"""
run_cli_agent.py — 方式三：CLI Agent（LLM + 白名单 subprocess）

默认 named：run_cli(command enum) → 拼 argv → subprocess（无 shell）。
可选 bash：沙箱黑名单 + 命令头白名单后再 shell=True。

用法：
  python src\\adapters\\cli\\run_cli_agent.py "北京天气"
  python src\\adapters\\cli\\run_cli_agent.py --mode named -q "茅台2023营收"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.common.config import CHAT_PROVIDERS, get_chat_client  # noqa: E402

BASE_DIR = _ROOT
CLI_MAIN = Path(__file__).resolve().parent / "main.py"
PY = sys.executable
FINCLI_ARGV = [PY, str(CLI_MAIN)]

NAMED_COMMANDS = {
    "rag_search": {
        "argv": FINCLI_ARGV + ["search"],
        "arg_map": {
            "query": "--query",
            "stock_code": "--stock-code",
            "year": "--year",
            "top_k": "--top-k",
        },
    },
    "rag_list_companies": {
        "argv": FINCLI_ARGV + ["list-companies"],
        "arg_map": {},
    },
    "weather": {
        "argv": FINCLI_ARGV + ["weather"],
        "arg_map": {"city": "--city"},
    },
}

DANGEROUS_PATTERNS = [
    r"\brm\b",
    r"\bdel\b",
    r"\brmdir\b",
    r"\bformat\b",
    r"\bmkfs\b",
    r"\bdd\b",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bsudo\b",
    r"\bcurl\b.*\|\s*sh",
    r"\bwget\b.*\|\s*sh",
    r"\bnc\b",
    r"\bnetcat\b",
    r"/etc/passwd",
    r"/etc/shadow",
]

ALLOWED_HEADS = {
    "python",
    "python3",
    "py",
    "git",
    "ls",
    "dir",
    "cat",
    "echo",
    "type",
}


def run_named(command: str, args: dict) -> str:
    """白名单 argv 拼接，禁止 shell=True。"""
    spec = NAMED_COMMANDS.get(command)
    if spec is None:
        return f"[run_cli] 未知命令：{command}（白名单：{list(NAMED_COMMANDS)}）"

    argv = list(spec["argv"])
    for key, flag in spec["arg_map"].items():
        val = args.get(key)
        if val is not None:
            argv.extend([flag, str(val)])

    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(BASE_DIR),
            env={**os.environ},
            shell=False,
        )
    except subprocess.TimeoutExpired:
        return "[run_cli] 命令执行超时（>60s）"
    if proc.returncode != 0:
        return f"[run_cli] 命令失败（code={proc.returncode}）：{proc.stderr[-500:]}"
    return proc.stdout


def sandbox_check(command: str) -> str | None:
    for pat in DANGEROUS_PATTERNS:
        if re.search(pat, command, re.IGNORECASE):
            return f"沙箱拦截：命中危险模式 {pat!r}"
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        return "沙箱拦截：命令解析失败"
    if not tokens:
        return "沙箱拦截：空命令"
    head = Path(tokens[0]).name.lower()
    # 允许本机 python 调 main.py；禁止任意可执行文件
    if head not in ALLOWED_HEADS:
        return f"沙箱拦截：{tokens[0]!r} 不在白名单 {sorted(ALLOWED_HEADS)} 中"
    return None


def run_bash(command: str) -> str:
    blocked = sandbox_check(command)
    if blocked:
        return f"[run_bash] {blocked}"
    try:
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=15,
            cwd=str(BASE_DIR),
            env={**os.environ},
        )
    except subprocess.TimeoutExpired:
        return "[run_bash] 命令执行超时（>15s）"
    out = proc.stdout
    if proc.returncode != 0:
        out += f"\n[run_bash] 退出码 {proc.returncode}，stderr：{proc.stderr[-300:]}"
    return out


NAMED_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "run_cli",
            "description": (
                "执行预批准的命令行工具。command 只能取白名单："
                "rag_search / rag_list_companies / weather。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": list(NAMED_COMMANDS.keys()),
                    },
                    "args": {
                        "type": "object",
                        "description": (
                            "rag_search: {query, stock_code?, year?, top_k?}; "
                            "weather: {city}"
                        ),
                    },
                },
                "required": ["command"],
            },
        },
    },
]

BASH_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "run_bash",
            "description": (
                "在沙箱中执行一条 shell 命令。可用："
                f"{PY} {CLI_MAIN} search --query '营收' --stock-code 600519 --year 2023；"
                f"{PY} {CLI_MAIN} list-companies；"
                f"{PY} {CLI_MAIN} weather --city 北京。"
                "危险命令会被拦截。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "完整 shell 命令"},
                },
                "required": ["command"],
            },
        },
    },
]

MODE_DISPATCH = {
    "named": (
        NAMED_TOOLS_SCHEMA,
        lambda args: run_named(args["command"], args.get("args") or {}),
    ),
    "bash": (BASH_TOOLS_SCHEMA, lambda args: run_bash(args["command"])),
}

SYSTEM_PROMPT_NAMED = (
    "你是一名金融分析助手。通过 run_cli 调用预批准命令查年报与天气。"
    "年报问题必须先 run_cli(command='rag_search', args={...})，只依据返回作答。"
    "知识库：贵州茅台(600519)/五粮液(000858)/宁德时代(300750)/海康威视(002415)/中国平安(601318)。"
    "rag_search 的 query 不要含公司名/年份。不在库内请明确告知。"
)

SYSTEM_PROMPT_BASH = (
    "你是一名金融分析助手。通过 run_bash 在沙箱执行 python 调用 cli/main.py。"
    "查年报、列公司、查天气。只依据命令输出作答，不要编造。"
)


def run(client, model: str, question: str, mode: str, verbose: bool = True) -> dict:
    tools_schema, executor = MODE_DISPATCH[mode]
    sys_prompt = SYSTEM_PROMPT_NAMED if mode == "named" else SYSTEM_PROMPT_BASH
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": question},
    ]
    t0 = time.time()
    tool_call_log: list[dict] = []

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools_schema,
        tool_choice="auto",
    )
    msg = resp.choices[0].message

    if msg.tool_calls:
        messages.append(msg)
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            tool_call_log.append({"name": tc.function.name, "args": args})
            if verbose:
                print(f"  → [{mode}] {tc.function.name}({args})")
            try:
                result = executor(args)
            except Exception as e:
                result = f"[{mode}] 执行异常：{e}"
            preview = (result or "")[:120].replace("\n", " ")
            if verbose:
                print(f"    ↩ {preview}{'...' if len(result or '') > 120 else ''}\n")
            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": result}
            )

        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools_schema,
            tool_choice="auto",
        )
        msg = resp.choices[0].message

    answer = msg.content or ""
    elapsed = time.time() - t0
    if verbose:
        print(f"  → [llm] 最终回答（{elapsed:.1f}s）")
    return {"answer": answer, "tool_calls": tool_call_log, "elapsed": elapsed}


DEMO_QUESTIONS = [
    "贵州茅台2023年营业收入是多少？",
    "北京天气怎么样？",
    "比亚迪2023年营收是多少？",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="V4 CLI Agent")
    parser.add_argument("positional_q", nargs="?", help="问题")
    parser.add_argument("--question", "-q")
    parser.add_argument("--mode", default="named", choices=["named", "bash"])
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
        print(f"[CLI/{args.mode}] provider={args.provider} model={model}\n", file=sys.stderr)

    q = args.question or args.positional_q
    questions = DEMO_QUESTIONS if args.demo else ([q] if q else [DEMO_QUESTIONS[0]])
    results = []
    for i, question in enumerate(questions, 1):
        if not args.json:
            print("=" * 60)
            print(f"Q{i}：{question}")
            print("=" * 60)
        result = run(
            client,
            model,
            question,
            args.mode,
            verbose=not (args.quiet or args.json),
        )
        result["question"] = question
        result["mode"] = args.mode
        results.append(result)
        if not args.json:
            print("\n最终回答：")
            print(result["answer"])
            print()

    if args.json:
        print(json.dumps(results[0] if len(results) == 1 else results, ensure_ascii=False))


if __name__ == "__main__":
    main()
