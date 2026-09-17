"""BM25 稀疏检索：jieba 分词 + BM25Okapi，语料来自 FAISS meta。"""

from __future__ import annotations

from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from src.rag.indexer import load_index

_BM25: BM25Okapi | None = None
_META: list[dict] | None = None


def _tokenize(text: str) -> list[str]:
    import jieba

    return [t for t in jieba.cut(text) if t.strip()]


def build_bm25(meta_list: list[dict] | None = None) -> tuple[BM25Okapi, list[dict]]:
    """用 meta 文本构建 BM25；缺省从 vectorstore 加载。"""
    if meta_list is None:
        _, meta_list = load_index()
    tokenized = [_tokenize(str(m.get("text") or "")) for m in meta_list]
    return BM25Okapi(tokenized), meta_list


def _ensure_loaded(
    meta_list: list[dict] | None = None,
) -> tuple[BM25Okapi, list[dict]]:
    global _BM25, _META
    if _BM25 is None or _META is None or meta_list is not None:
        _BM25, _META = build_bm25(meta_list)
    return _BM25, _META


def search(
    query: str,
    k: int = 5,
    *,
    meta_list: list[dict] | None = None,
    bm25: BM25Okapi | None = None,
) -> list[dict[str, Any]]:
    """返回 TopK hits：meta 字段 + bm25_score；score 同步为 bm25_score。"""
    if k <= 0:
        return []

    if bm25 is None or meta_list is None:
        bm25, meta_list = _ensure_loaded(meta_list)

    tokens = _tokenize(query)
    if not tokens:
        return []

    scores = bm25.get_scores(tokens)
    top_idx = np.argsort(scores)[::-1][:k]

    hits: list[dict[str, Any]] = []
    for idx in top_idx:
        score = float(scores[idx])
        if score < 1e-9:
            continue
        if idx < 0 or idx >= len(meta_list):
            continue
        hit = dict(meta_list[idx])
        hit["bm25_score"] = score
        hit["score"] = score
        hits.append(hit)
    return hits
