"""本地 CrossEncoder 精排；失败时跳过并保持原序。"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

# 确保在 import torch / CrossEncoder 前已放行双 OpenMP（config 也会设，此处兜底）
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from src.common.config import RERANK_MODEL_PATH, RERANK_TOP_K

logger = logging.getLogger(__name__)

_RERANKER = None


def _resolve_model_path(model: str | Path | None = None) -> Path:
    if model is None:
        return Path(RERANK_MODEL_PATH)
    path = Path(model)
    if path.exists():
        return path
    # 允许传相对 models/ 的目录名
    from src.common.config import MODELS_DIR

    candidate = MODELS_DIR / str(model)
    return candidate if candidate.exists() else path


def _get_reranker(model: str | Path | None = None):
    """懒加载并缓存 CrossEncoder。"""
    global _RERANKER
    if _RERANKER is not None and model is None:
        return _RERANKER

    from sentence_transformers import CrossEncoder

    model_path = _resolve_model_path(model)
    if not model_path.exists():
        raise FileNotFoundError(
            f"本地 Rerank 模型不存在: {model_path}\n"
            f"请将 BAAI/bge-reranker-base 下载到该目录后重试。"
        )
    reranker = CrossEncoder(str(model_path))
    if model is None:
        _RERANKER = reranker
    return reranker


def rerank(
    query: str,
    hits: list[dict[str, Any]],
    top_k: int | None = None,
    *,
    model: str | Path | None = None,
) -> list[dict[str, Any]]:
    """对候选 hits 精排；模型缺失或推理失败则截断返回原序。"""
    top_k = RERANK_TOP_K if top_k is None else top_k
    if not hits or top_k <= 0:
        return []

    try:
        reranker = _get_reranker(model)
        pairs = [(query, str(h.get("text") or "")) for h in hits]
        scores = reranker.predict(pairs)
        ranked = []
        for hit, score in zip(hits, scores):
            item = dict(hit)
            item["rerank_score"] = float(score)
            item["score"] = float(score)
            ranked.append(item)
        ranked.sort(key=lambda x: -x.get("rerank_score", 0.0))
        return ranked[:top_k]
    except ImportError:
        logger.warning(
            "sentence-transformers 未安装，跳过 Rerank"
            "（pip install sentence-transformers）"
        )
        return hits[:top_k]
    except Exception as e:
        logger.warning("Rerank 失败，使用 RRF 原序: %s", e)
        return hits[:top_k]
