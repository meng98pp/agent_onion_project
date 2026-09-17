"""V3 评估：默认 Hit@K + 消融；可选 --ragas 四指标。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.common.config import (
    EMBED_DIM,
    EMBEDDING_MODEL,
    EVAL_GOLD_PATH,
    EVAL_RESULTS_DIR,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
)
from src.rag.ask import ask

ModeName = Literal["vector_only", "hybrid", "hybrid_rerank"]


def load_gold(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or EVAL_GOLD_PATH
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"金标应为 JSON 数组: {path}")
    return data


def hit_at_k(hits: list[dict[str, Any]], must_substr: str) -> bool:
    """TopK 的 text 或 source 是否包含必须子串。"""
    needle = (must_substr or "").strip()
    if not needle:
        return False
    for h in hits:
        blob = f"{h.get('text') or ''} {h.get('source') or ''}"
        if needle in blob:
            return True
    return False


def run_mode(
    questions: list[dict[str, Any]],
    mode: ModeName,
    *,
    k: int = 5,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    hits_flags: list[bool] = []

    for i, q in enumerate(questions, 1):
        question = str(q.get("question") or "")
        must = str(q.get("must_cite_source_substring") or "")
        qtype = str(q.get("type") or "")
        print(f"[{mode}] ({i}/{len(questions)}) {question[:40]}...", flush=True)
        try:
            result = ask(question, k=k, mode=mode)
            contexts = [str(h.get("text") or "") for h in result["hits"]]
            hit = hit_at_k(result["hits"], must)
            hits_flags.append(hit)
            rows.append(
                {
                    "question": question,
                    "ground_truth": q.get("ground_truth", ""),
                    "must_cite_source_substring": must,
                    "question_type": qtype,
                    "answer": result["answer"],
                    "contexts": contexts,
                    "hit": hit,
                    "citations": result.get("citations") or [],
                    "error": None,
                }
            )
        except Exception as e:
            hits_flags.append(False)
            rows.append(
                {
                    "question": question,
                    "ground_truth": q.get("ground_truth", ""),
                    "must_cite_source_substring": must,
                    "question_type": qtype,
                    "answer": "",
                    "contexts": [],
                    "hit": False,
                    "citations": [],
                    "error": str(e),
                }
            )

    hit_rate = (sum(hits_flags) / len(hits_flags)) if hits_flags else 0.0
    return {
        "mode": mode,
        "k": k,
        "n": len(questions),
        "hit_at_k": hit_rate,
        "hit_count": sum(hits_flags),
        "rows": rows,
    }


def maybe_run_ragas(rows: list[dict[str, Any]]) -> dict[str, float] | None:
    """可选 RAGAS 四指标；用 DashScope，不依赖 OPENAI_API_KEY。"""
    import os
    import warnings

    try:
        from datasets import Dataset
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas import evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper

        # ragas 0.4 仍可用旧入口；collections 需 Instructor LLM，与 week10 写法不一致
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Importing .* from 'ragas.metrics' is deprecated.*",
                category=DeprecationWarning,
            )
            from ragas.metrics import (
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
            )
    except ImportError as e:
        print(f"跳过 RAGAS（请 pip install ragas datasets langchain-openai）: {e}", flush=True)
        return None

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("跳过 RAGAS：缺少 DASHSCOPE_API_KEY", flush=True)
        return None
    # 部分 openai/langchain 嵌套客户端仍读 OPENAI_API_KEY
    os.environ.setdefault("OPENAI_API_KEY", api_key)

    # 拒答题的 ground_truth 是说明文字，不适合四指标
    usable = [
        r
        for r in rows
        if r.get("answer")
        and r.get("contexts") is not None
        and not r.get("error")
        and r.get("question_type") != "should_refuse"
    ]
    if not usable:
        print("跳过 RAGAS：无有效样本", flush=True)
        return None

    ragas_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=OPENAI_MODEL,
            api_key=api_key,
            base_url=OPENAI_BASE_URL,
            temperature=0,
        )
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            api_key=api_key,
            base_url=OPENAI_BASE_URL,
            dimensions=EMBED_DIM,
            # DashScope 兼容接口只要原始字符串，关闭长度检查避免 400
            check_embedding_ctx_length=False,
        )
    )
    for metric in [faithfulness, answer_relevancy, context_precision, context_recall]:
        metric.llm = ragas_llm
        metric.embeddings = ragas_embeddings

    dataset = Dataset.from_list(
        [
            {
                "question": r["question"],
                "answer": r["answer"],
                "contexts": r["contexts"],
                "ground_truth": r.get("ground_truth") or "",
            }
            for r in usable
        ]
    )

    try:
        score = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
    except Exception as e:
        print(f"RAGAS 评估失败: {e}", flush=True)
        return None

    import numpy as np

    def _mean(metric_name: str) -> float:
        vals = [
            v
            for v in score[metric_name]
            if v is not None and not (isinstance(v, float) and np.isnan(v))
        ]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "faithfulness": _mean("faithfulness"),
        "answer_relevancy": _mean("answer_relevancy"),
        "context_precision": _mean("context_precision"),
        "context_recall": _mean("context_recall"),
    }


def save_result(payload: dict[str, Any], *, tag: str) -> Path:
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = EVAL_RESULTS_DIR / f"{tag}_{ts}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入 {out}", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Hit@K / 消融 / 可选 RAGAS")
    parser.add_argument(
        "--ablation",
        default="hybrid",
        help="逗号分隔模式：vector_only,hybrid,hybrid_rerank",
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--gold",
        type=Path,
        default=None,
        help="金标 JSON 路径，默认 evaluation/gold_qa.json",
    )
    parser.add_argument(
        "--ragas",
        action="store_true",
        help="额外跑 RAGAS 四指标（需已安装 ragas/datasets，且会调 LLM）",
    )
    args = parser.parse_args()

    modes = [m.strip() for m in args.ablation.split(",") if m.strip()]
    allowed: set[str] = {"vector_only", "hybrid", "hybrid_rerank"}
    for m in modes:
        if m not in allowed:
            raise SystemExit(f"未知 mode: {m!r}，可选 {sorted(allowed)}")

    questions = load_gold(args.gold)
    summary: list[dict[str, Any]] = []

    for mode in modes:
        result = run_mode(questions, mode=mode, k=args.k)  # type: ignore[arg-type]
        ragas_scores = None
        if args.ragas:
            ragas_scores = maybe_run_ragas(result["rows"])
            result["ragas"] = ragas_scores

        print(
            f"── {mode} Hit@{args.k} = {result['hit_at_k']:.3f} "
            f"({result['hit_count']}/{result['n']})",
            flush=True,
        )
        if ragas_scores:
            for name, val in ragas_scores.items():
                print(f"   RAGAS {name}: {val:.4f}", flush=True)

        save_result(result, tag=mode)
        summary.append(
            {
                "mode": mode,
                "hit_at_k": result["hit_at_k"],
                "hit_count": result["hit_count"],
                "n": result["n"],
                "ragas": ragas_scores,
            }
        )

    if len(summary) > 1:
        save_result({"ablation": summary}, tag="ablation_summary")


if __name__ == "__main__":
    main()
