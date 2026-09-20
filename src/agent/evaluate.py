"""对照评估：手写 Prompt 解析 vs Function Calling。"""

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
from src.common.config import MAX_STEPS, OUTPUT_DIR  # noqa: E402

EVAL_QUESTIONS = [
    {
        "id": "Q1",
        "question": "贵州茅台和五粮液2023年的毛利率哪家更高？差多少个百分点？",
        "expected_tools": ["company_lookup", "company_lookup", "financial_indicator", "financial_indicator", "calculator"],
        "note": "跨公司对比 + 计算",
    },
    {
        "id": "Q2",
        "question": "宁德时代2021年到2023年的营业收入分别是多少？复合增长率是多少？",
        "expected_tools": ["company_lookup", "financial_indicator", "calculator"],
        "note": "跨年度趋势 + CAGR",
    },
    {
        "id": "Q3",
        "question": "海康威视2023年年报中提到了哪些主要风险因素？",
        "expected_tools": ["rag_search"],
        "note": "单次检索，定性问题",
    },
    {
        "id": "Q4",
        "question": "贵州茅台过去一年（2023年全年）的股价涨跌幅是多少？",
        "expected_tools": ["company_lookup", "stock_price"],
        "note": "lookup 后再查行情",
    },
    {
        "id": "Q5",
        "question": "请预测明年A股市场的走势",
        "expected_tools": [],
        "note": "超出能力边界，应拒绝作答",
    },
]


def _run_single(mode: str, question: str, max_steps: int) -> dict:
    if mode == "manual":
        from src.agent.react_manual import run as react_run
    else:
        from src.agent.react_fc import run as react_run

    steps: list[dict] = []
    parse_errors = 0
    start = time.time()
    final_answer = None
    success = False

    for step_data in react_run(question, max_steps=max_steps):
        steps.append(step_data)
        if step_data["type"] in ("unparseable", "error") and "格式解析失败" in str(
            step_data.get("observation") or ""
        ):
            parse_errors += 1
        if step_data["type"] == "final":
            final_answer = step_data.get("answer")
            success = True

    action_steps = [s for s in steps if s["type"] == "action"]
    return {
        "mode": mode,
        "total_steps": len(action_steps),
        "elapsed_s": round(time.time() - start, 1),
        "success": success,
        "parse_errors": parse_errors,
        "tools_used": [s.get("action") for s in action_steps],
        "final_answer": (final_answer or "")[:300],
    }


def evaluate(output_path: Path | None = None, max_steps: int | None = None) -> list[dict]:
    limit = max_steps if max_steps is not None else MAX_STEPS
    results: list[dict] = []

    for q in EVAL_QUESTIONS:
        print(f"\n{'-' * 60}")
        print(f"[{q['id']}] {q['question']}")
        print(f"  预期工具: {q['expected_tools']}  ({q['note']})")
        for mode in ("manual", "fc"):
            print(f"  运行 [{mode}]...", end=" ", flush=True)
            r = _run_single(mode, q["question"], limit)
            print(f"步骤:{r['total_steps']}  耗时:{r['elapsed_s']}s  成功:{r['success']}")
            results.append({**q, **r})

    print(f"\n{'=' * 60}")
    print(f"{'ID':<4} {'Mode':<8} {'Steps':<7} {'Time(s)':<9} {'Success':<8} {'ParseErr'}")
    print("-" * 60)
    for r in results:
        print(
            f"{r['id']:<4} {r['mode']:<8} {r['total_steps']:<7} "
            f"{r['elapsed_s']:<9} {str(r['success']):<8} {r['parse_errors']}"
        )

    for mode in ("manual", "fc"):
        mode_results = [r for r in results if r["mode"] == mode]
        n = len(mode_results) or 1
        avg_steps = sum(r["total_steps"] for r in mode_results) / n
        avg_time = sum(r["elapsed_s"] for r in mode_results) / n
        success_rate = sum(r["success"] for r in mode_results) / n
        total_errors = sum(r["parse_errors"] for r in mode_results)
        print(
            f"\n[{mode}] 平均步数:{avg_steps:.1f}  平均耗时:{avg_time:.1f}s  "
            f"成功率:{success_rate:.0%}  解析错误总数:{total_errors}"
        )

    out = output_path or (OUTPUT_DIR / "evaluate_v5.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存至 {out}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="V5 manual vs fc 对照评估")
    parser.add_argument("--output", default=None, help="保存 JSON 路径")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()
    path = Path(args.output) if args.output else None
    evaluate(path, args.max_steps)


if __name__ == "__main__":
    main()
