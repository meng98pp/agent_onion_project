"""
compare.py — 同一问题 × Function Call / MCP / CLI 对比

以子进程跑各适配器的 --json 输出，写入 output/compare_result.md。

用法（在仓库根执行）：
  python src\\compare.py
  python src\\compare.py --provider dashscope
  python src\\compare.py --questions "茅台2024营收" "北京天气"
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# 本文件在 src/compare.py；适配器与 .env 都以仓库根为准，不能再拼一层 src/
ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

MODES = [
    (
        "Function Call",
        [
            PY,
            str(ROOT / "src" / "adapters" / "function_call" / "run_fc.py"),
            "--json",
            "--quiet",
        ],
    ),
    (
        "MCP",
        [
            PY,
            str(ROOT / "src" / "adapters" / "mcp" / "run_mcp_host.py"),
            "--json",
            "--quiet",
        ],
    ),
    (
        "CLI(named)",
        [
            PY,
            str(ROOT / "src" / "adapters" / "cli" / "run_cli_agent.py"),
            "--mode",
            "named",
            "--json",
            "--quiet",
        ],
    ),
]

DEFAULT_QUESTIONS = [
    "贵州茅台2024年营业收入是多少？",
    "宁德时代2023年营收和净利润是多少？另外总部宁德的天气如何？",
    "对比贵州茅台和五粮液2023年的营收。",
    "比亚迪2024年营收是多少？",
]

REFUSE_PATTERNS = ["不在", "未收录", "无法", "没有收录", "不在库", "不在知识库", "未能", "查不到"]


def run_one(mode_cmd: list[str], question: str, provider: str) -> dict:
    cmd = mode_cmd + ["--provider", provider, "-q", question]
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180,
            cwd=str(ROOT),
            env={**os.environ},
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "超时(>180s)", "elapsed": time.time() - t0}

    wall = time.time() - t0
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        # 异常类型在 traceback 末尾；取头会只看到 send_raw_request 这种中段
        return {"ok": False, "error": err[-800:], "elapsed": wall}

    out = proc.stdout.strip().splitlines()
    if not out:
        return {"ok": False, "error": "无输出：" + (proc.stderr or "")[-300:], "elapsed": wall}
    try:
        data = json.loads(out[-1])
    except json.JSONDecodeError as e:
        return {"ok": False, "error": f"JSON 解析失败：{e}", "elapsed": wall}

    data["ok"] = True
    data["wall_elapsed"] = wall
    return data


def summarize(data: dict, question: str) -> dict:
    if not data.get("ok"):
        return {
            "tools": "-",
            "tool_count": 0,
            "llm_elapsed": "-",
            "answer_preview": "(失败) " + (data.get("error", "")[:180]).replace("\n", " "),
            "refused": None,
        }
    tcs = data.get("tool_calls") or []
    tool_names = ", ".join(t["name"] for t in tcs) or "(无工具调用)"
    answer = data.get("answer") or ""
    need_refuse = "比亚迪" in question
    refused = any(p in answer for p in REFUSE_PATTERNS) if need_refuse else None
    return {
        "tools": tool_names,
        "tool_count": len(tcs),
        "llm_elapsed": f"{data.get('elapsed', 0):.1f}s",
        "answer_preview": answer[:80].replace("\n", " ") + ("..." if len(answer) > 80 else ""),
        "refused": refused,
    }


def run_compare(questions: list[str], provider: str) -> list[dict]:
    rows = []
    for qi, q in enumerate(questions, 1):
        print(f"\n{'=' * 70}\nQ{qi}：{q}\n{'=' * 70}")
        for mode_name, mode_cmd in MODES:
            print(f"  ▶ {mode_name} ...", end=" ", flush=True)
            data = run_one(mode_cmd, q, provider)
            s = summarize(data, q)
            status = "✓" if data.get("ok") else "✗"
            print(f"{status} 工具[{s['tool_count']}] {s['llm_elapsed']}")
            rows.append({"question": q, "mode": mode_name, **s, "raw_ok": data.get("ok", False)})
    return rows


def write_markdown(rows: list[dict], questions: list[str], provider: str, path: Path) -> None:
    lines = [
        "# 三方式对比结果（Function Call / MCP / CLI）",
        "",
        f"- LLM provider：`{provider}`",
        "- 生成时间：由 `python src\\compare.py` 实跑生成",
        f"- 问题数：{len(questions)}，方式数：{len(MODES)}",
        "",
        "## 对比表",
        "",
        "| 问题 | 方式 | 工具调用 | 工具数 | LLM耗时 | 正确拒绝幻觉 | 答案摘要 |",
        "|------|------|---------|:------:|:-------:|:------------:|---------|",
    ]
    for r in rows:
        refuse_cell = (
            "-"
            if r["refused"] is None
            else ("✓ 拒绝" if r["refused"] else "✗ 未拒绝(可能幻觉)")
        )
        lines.append(
            f"| {r['question']} | {r['mode']} | {r['tools']} | {r['tool_count']} | "
            f"{r['llm_elapsed']} | {refuse_cell} | {r['answer_preview']} |"
        )

    lines += [
        "",
        "## 解读",
        "",
        "- **能力一致**：同一业务后端，差异在接入方式而非检索算法。",
        "- **接入成本**：FC 手写 schema；MCP 自动 list_tools；CLI 白名单子命令。",
        "- **安全**：CLI(named) 禁止任意 shell；bash 形态需沙箱（本 compare 默认 named）。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="V4 三方式对比")
    parser.add_argument("--questions", nargs="+", default=DEFAULT_QUESTIONS)
    parser.add_argument("--provider", default="dashscope", choices=["deepseek", "dashscope"])
    args = parser.parse_args()

    print(f"[compare] provider={args.provider}, {len(args.questions)} 题 × {len(MODES)} 方式\n")
    rows = run_compare(args.questions, args.provider)

    out_dir = ROOT / "output"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "compare_result.md"
    write_markdown(rows, args.questions, args.provider, out_path)

    print(f"\n{'=' * 70}\n对比表已写入 {out_path}\n{'=' * 70}")
    print(f"{'问题':<28}{'方式':<16}{'工具数':<6}{'LLM耗时':<10}{'拒绝':<8}")
    print("-" * 70)
    for r in rows:
        refuse = "-" if r["refused"] is None else ("✓" if r["refused"] else "✗")
        print(
            f"{r['question'][:26]:<28}{r['mode']:<16}{r['tool_count']:<6}"
            f"{r['llm_elapsed']:<10}{refuse:<8}"
        )


if __name__ == "__main__":
    main()
