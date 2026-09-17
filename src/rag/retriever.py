"""在线向量检索：embed query → FAISS TopK。"""

from __future__ import annotations

from typing import Any

from openai import OpenAI

from src.common.config import get_openai_client
from src.rag.embedder import embed_query
from src.rag.indexer import load_index

_INDEX = None
_META: list[dict] | None = None


def _ensure_loaded():
    global _INDEX, _META
    if _INDEX is None or _META is None:
        _INDEX, _META = load_index()
    return _INDEX, _META


def search(
    query: str,
    k: int = 5,
    *,
    client: OpenAI | None = None,
    index=None,
    meta_list: list[dict] | None = None,
) -> list[dict[str, Any]]:
    """返回 TopK hits：meta 字段 + score（内积≈余弦）。"""
    if k <= 0:
        return []

    if index is None or meta_list is None:
        index, meta_list = _ensure_loaded()

    client = client or get_openai_client()
    query_vec = embed_query(query, client=client)
    scores, indices = index.search(query_vec, min(k, index.ntotal or k))

    hits: list[dict[str, Any]] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(meta_list):
            continue
        hit = dict(meta_list[idx])
        hit["score"] = float(score)
        hits.append(hit)
    return hits
