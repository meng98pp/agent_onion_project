"""V2 在线编排：问题 → TopK → Prompt → 答案+引用。禁止写 FAISS 细节。"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.rag.generator import answer
from src.rag.retriever import search


def ask(question: str, k: int = 5) -> dict:
    hits = search(question, k=k)
    text, cites = answer(question, hits)
    return {"answer": text, "citations": cites, "hits": hits}


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "这份文档的主题是什么？"
    result = ask(q)
    print(result["answer"])
    if result["citations"]:
        print("\n── 来源 ──")
        for c in result["citations"]:
            print(f"  {c.get('label', c)}")

