"""验收：假向量 FAISS TopK 命中；低分拒答不调 LLM。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.common.config import EMBED_DIM, SCORE_THRESHOLD
from src.rag.generator import REFUSAL, answer
from src.rag.retriever import search

faiss = pytest.importorskip("faiss")


def _normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(norms, 1e-9)


@pytest.fixture
def tiny_store():
    """3 条正交倾向向量 + 元数据。"""
    # 构造近似可区分的单位向量
    raw = np.eye(3, EMBED_DIM, dtype="float32")
    embeddings = _normalize(raw)
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(embeddings)

    meta = [
        {
            "chunk_id": "demo_semantic_0000",
            "text": "2023年毛利率为45.2%。",
            "source": "demo.txt",
            "page_num": 1,
            "section_path": "二、盈利能力",
            "strategy": "semantic",
            "parent_id": None,
            "block_types": ["text"],
            "stock_code": "",
            "year": "2023",
        },
        {
            "chunk_id": "demo_semantic_0001",
            "text": "本公司主要从事示例消费品。",
            "source": "demo.txt",
            "page_num": 2,
            "section_path": "一、主营业务概述",
            "strategy": "semantic",
            "parent_id": None,
            "block_types": ["text"],
            "stock_code": "",
            "year": "2023",
        },
        {
            "chunk_id": "demo_semantic_0002",
            "text": "营业收入 120.5亿元。",
            "source": "demo.txt",
            "page_num": 3,
            "section_path": "五、主要财务数据",
            "strategy": "semantic",
            "parent_id": None,
            "block_types": ["table"],
            "stock_code": "",
            "year": "2023",
        },
    ]
    return index, meta, embeddings


def test_search_topk_hits_nearest_chunk(tiny_store):
    index, meta, embeddings = tiny_store
    query_vec = embeddings[0:1].copy()  # 应命中毛利率那条

    with patch("src.rag.retriever.embed_query", return_value=query_vec):
        hits = search("毛利率", k=2, client=MagicMock(), index=index, meta_list=meta)

    assert len(hits) == 2
    assert hits[0]["chunk_id"] == "demo_semantic_0000"
    assert "毛利率" in hits[0]["text"]
    assert "score" in hits[0]
    assert hits[0]["score"] >= hits[1]["score"]
    assert hits[0]["source"] == "demo.txt"
    assert hits[0]["page_num"] == 1


def test_answer_refuses_when_score_below_threshold():
    hits = [
        {
            "chunk_id": "x",
            "text": "无关内容",
            "source": "demo.txt",
            "page_num": 1,
            "score": SCORE_THRESHOLD - 0.01,
        }
    ]
    text, cites = answer("不存在的公司XYZ的营收", hits, client=MagicMock())
    assert text == REFUSAL
    assert cites == []


def test_answer_refuses_on_empty_hits():
    text, cites = answer("任意问题", [], client=MagicMock())
    assert text == REFUSAL
    assert cites == []
