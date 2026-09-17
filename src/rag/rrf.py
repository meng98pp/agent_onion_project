"""Reciprocal Rank Fusion：纯函数，无 I/O。"""

from __future__ import annotations


def rrf_fuse(rank_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """按排名融合多路召回；同分按 doc_id 稳定排序。"""
    scores: dict[str, float] = {}
    for ranks in rank_lists:
        for i, doc_id in enumerate(ranks):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + i + 1)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))
