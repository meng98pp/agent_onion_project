"""混合检索：向量 + BM25 → RRF → 可选 Rerank。"""

from __future__ import annotations

from typing import Any

from openai import OpenAI

from src.common.config import RETRIEVE_TOP_N, RRF_K
from src.rag import bm25_index
from src.rag.reranker import rerank
from src.rag.retriever import search as vector_search
from src.rag.rrf import rrf_fuse


def _score_key(hit: dict[str, Any], *fields: str) -> float:
    for f in fields:
        if hit.get(f) is not None:
            return float(hit[f])
    return float(hit.get("score") or 0.0)


def _print_ranked(
    title: str,
    hits: list[dict[str, Any]],
    *,
    score_fields: tuple[str, ...] = ("score",),
) -> None:
    """按分值降序打印，便于定位对比两路召回。"""
    ranked = sorted(hits, key=lambda h: _score_key(h, *score_fields), reverse=True)
    print(f"\n── {title}（按分值降序，共 {len(ranked)} 条）──", flush=True)
    for i, h in enumerate(ranked, 1):
        score = _score_key(h, *score_fields)
        cid = h.get("chunk_id", "")
        source = h.get("source", "")
        page = h.get("page_num", "?")
        text = (h.get("text") or "").replace("\n", " ").strip()
        preview = text[:80] + ("…" if len(text) > 80 else "")
        extras = []
        if h.get("vector_score") is not None:
            extras.append(f"vec={float(h['vector_score']):.4f}")
        if h.get("bm25_score") is not None:
            extras.append(f"bm25={float(h['bm25_score']):.4f}")
        if h.get("rrf_score") is not None:
            extras.append(f"rrf={float(h['rrf_score']):.6f}")
        if h.get("rerank_score") is not None:
            extras.append(f"rerank={float(h['rerank_score']):.4f}")
        extra_s = (" | " + " ".join(extras)) if extras else ""
        print(
            f"  [{i:02d}] score={score:.6f}{extra_s} | {cid} | {source}:p.{page}\n"
            f"       {preview}",
            flush=True,
        )


def search_hybrid(
    query: str,
    k: int = 5,
    *,
    use_rerank: bool = False,
    retrieve_n: int | None = None,
    client: OpenAI | None = None,
    index=None,
    meta_list: list[dict] | None = None,
    debug: bool = False,
) -> list[dict[str, Any]]:
    """向量与 BM25 各召回 retrieve_n，RRF 融合后取 TopK；可选精排。"""
    if k <= 0:
        return []

    n = retrieve_n if retrieve_n is not None else max(RETRIEVE_TOP_N, k)

    vec_hits = vector_search(
        query, k=n, client=client, index=index, meta_list=meta_list
    )
    bm25_hits = bm25_index.search(query, k=n, meta_list=meta_list)

    if debug:
        _print_ranked("向量召回", vec_hits, score_fields=("score", "vector_score"))
        _print_ranked("BM25 召回", bm25_hits, score_fields=("bm25_score", "score"))

    vec_ids = [str(h.get("chunk_id") or "") for h in vec_hits if h.get("chunk_id")]
    bm25_ids = [str(h.get("chunk_id") or "") for h in bm25_hits if h.get("chunk_id")]

    chunk_map: dict[str, dict[str, Any]] = {}
    for h in vec_hits:
        cid = str(h.get("chunk_id") or "")
        if not cid:
            continue
        hit = dict(h)
        hit["vector_score"] = h.get("score")
        chunk_map[cid] = hit
    for h in bm25_hits:
        cid = str(h.get("chunk_id") or "")
        if not cid:
            continue
        if cid in chunk_map:
            chunk_map[cid]["bm25_score"] = h.get("bm25_score")
        else:
            chunk_map[cid] = dict(h)

    fused = rrf_fuse([vec_ids, bm25_ids], k=RRF_K)
    hits: list[dict[str, Any]] = []
    for cid, rrf_score in fused:
        if cid not in chunk_map:
            continue
        hit = dict(chunk_map[cid])
        hit["rrf_score"] = float(rrf_score)
        hit["score"] = float(rrf_score)
        hits.append(hit)

    if debug:
        _print_ranked("RRF 融合", hits, score_fields=("rrf_score", "score"))

    if use_rerank:
        final = rerank(query, hits, top_k=k)
        if debug:
            _print_ranked("Rerank 精排", final, score_fields=("rerank_score", "score"))
        return final
    return hits[:k]
