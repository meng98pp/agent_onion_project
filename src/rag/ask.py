"""V3 在线编排：vector_only / hybrid / hybrid_rerank → 答案+引用。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.rag.generator import answer
from src.rag.hybrid_retriever import search_hybrid
from src.rag.retriever import search

Mode = Literal["vector_only", "hybrid", "hybrid_rerank"]


def ask(
    question: str,
    k: int = 5,
    mode: Mode = "vector_only",
    *,
    debug: bool = False,
) -> dict:
    if mode == "vector_only":
        hits = search(question, k=k)
        text, cites = answer(question, hits)
    elif mode == "hybrid":
        hits = search_hybrid(question, k=k, use_rerank=False, debug=debug)
        text, cites = answer(question, hits, score_threshold=0.0)
    elif mode == "hybrid_rerank":
        hits = search_hybrid(question, k=k, use_rerank=True, debug=debug)
        text, cites = answer(question, hits, score_threshold=0.0)
    else:
        raise ValueError(f"未知 mode: {mode!r}，可选 vector_only / hybrid / hybrid_rerank")

    return {"answer": text, "citations": cites, "hits": hits, "mode": mode}


def main() -> None:
    parser = argparse.ArgumentParser(description="FinAgent RAG 问答（V3）")
    parser.add_argument(
        "--mode",
        choices=["vector_only", "hybrid", "hybrid_rerank"],
        default="vector_only",
        help="检索模式",
    )
    parser.add_argument("--q", "--query", dest="query", default=None, help="问题")
    parser.add_argument("-k", type=int, default=5, help="返回条数")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="打印向量/BM25/RRF(/Rerank) 按分值排序的检索结果",
    )
    parser.add_argument(
        "positional_q",
        nargs="?",
        default=None,
        help="兼容旧用法：直接传问题字符串",
    )
    args = parser.parse_args()

    q = args.query or args.positional_q or "这份文档的主题是什么？"
    result = ask(q, k=args.k, mode=args.mode, debug=args.debug)
    print(result["answer"])
    if result["citations"]:
        print("\n── 来源 ──")
        for c in result["citations"]:
            print(f"  {c.get('label', c)}")


if __name__ == "__main__":
    main()
