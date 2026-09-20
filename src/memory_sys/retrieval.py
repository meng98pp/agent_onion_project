"""Layer 4：FAISS 向量 0.7 + FTS5/BM25 0.3 并集。"""

from __future__ import annotations

from src.common.config import MEMORY_BM25_WEIGHT, MEMORY_VEC_WEIGHT, MEMORY_RETRIEVE_TOP_K
from src.memory_sys.fts_store import FTSStore
from src.memory_sys.vector_store import VectorStore


class HybridRetriever:
    def __init__(
        self,
        vs: VectorStore,
        fts: FTSStore,
        vec_weight: float = MEMORY_VEC_WEIGHT,
        bm25_weight: float = MEMORY_BM25_WEIGHT,
    ) -> None:
        self.vs = vs
        self.fts = fts
        self.vec_weight = vec_weight
        self.bm25_weight = bm25_weight

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        k = MEMORY_RETRIEVE_TOP_K if top_k is None else top_k
        vec_results = self.vs.search(query, top_k=k)
        bm25_results = self.fts.search(query, top_k=k) if self.fts.available else []
        if not vec_results and not bm25_results:
            return []
        if not bm25_results:
            for item in vec_results:
                item["source"] = "vector"
            return vec_results[:k]
        if not vec_results:
            for item in bm25_results:
                item["source"] = "bm25"
            return bm25_results[:k]

        merged: dict[str, dict] = {}
        for item in vec_results:
            key = str(item.get("id") or item.get("title") or "")
            merged[key] = {
                "id": item.get("id"),
                "title": item.get("title", ""),
                "content": item.get("content", ""),
                "category": item.get("category", ""),
                "vec_score": float(item.get("score", 0.0)),
                "bm25_score": 0.0,
            }
        for item in bm25_results:
            key = str(item.get("id") or item.get("title") or "")
            if key in merged:
                merged[key]["bm25_score"] = float(item.get("score", 0.0))
            else:
                merged[key] = {
                    "id": item.get("id"),
                    "title": item.get("title", ""),
                    "content": item.get("content", ""),
                    "category": item.get("category", ""),
                    "vec_score": 0.0,
                    "bm25_score": float(item.get("score", 0.0)),
                }
        for item in merged.values():
            item["score"] = (
                item["vec_score"] * self.vec_weight + item["bm25_score"] * self.bm25_weight
            )
            if item["vec_score"] > 0 and item["bm25_score"] > 0:
                item["source"] = "both"
            elif item["vec_score"] > 0:
                item["source"] = "vector"
            else:
                item["source"] = "bm25"
        ranked = sorted(merged.values(), key=lambda row: row["score"], reverse=True)
        return ranked[:k]
