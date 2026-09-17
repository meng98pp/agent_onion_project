"""RRF 纯函数：两路排名融合。"""

from __future__ import annotations

from src.rag.rrf import rrf_fuse


def test_rrf_fuses_two_rank_lists():
    # A 在向量第 1、BM25 第 2；B 仅向量第 2；C 仅 BM25 第 1
    fused = rrf_fuse(
        [
            ["A", "B"],
            ["C", "A"],
        ],
        k=60,
    )
    ids = [doc_id for doc_id, _ in fused]
    scores = dict(fused)

    assert ids[0] == "A"  # 两路都命中，分最高
    assert set(ids) == {"A", "B", "C"}
    assert scores["A"] == 1.0 / (60 + 1) + 1.0 / (60 + 2)
    assert scores["B"] == 1.0 / (60 + 2)
    assert scores["C"] == 1.0 / (60 + 1)


def test_rrf_empty_lists():
    assert rrf_fuse([]) == []
    assert rrf_fuse([[], []]) == []


def test_rrf_duplicate_in_same_list_still_accumulates():
    # 同一列表重复 id：按排名各加一次（调用方应避免，但函数行为确定）
    fused = rrf_fuse([["A", "A"]], k=60)
    assert len(fused) == 1
    assert fused[0][0] == "A"
    assert fused[0][1] == 1.0 / 61 + 1.0 / 62
