"""DashScope text-embedding-v3：批量并发 embed + L2 归一化。"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from openai import OpenAI

from src.common.config import (
    BATCH_SIZE,
    EMBED_CONCURRENCY,
    EMBED_DIM,
    EMBED_MAX_CHARS,
    EMBEDDING_MODEL,
    get_openai_client,
)


def _l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    return embeddings / norms


def sanitize_embed_text(text: str, *, max_chars: int = EMBED_MAX_CHARS) -> str:
    """满足 API [1, max_chars]：空串占位，超长截断。"""
    cleaned = (text or "").strip()
    if not cleaned:
        return "."
    if len(cleaned) > max_chars:
        return cleaned[:max_chars]
    return cleaned


def _embed_batch(client: OpenAI, batch: list[str]) -> list[list[float]]:
    """单批请求，失败重试 3 次。"""
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            resp = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch,
                dimensions=EMBED_DIM,
            )
            ordered = sorted(resp.data, key=lambda item: item.index)
            return [item.embedding for item in ordered]
        except Exception as exc:  # noqa: BLE001 — 重试后上抛
            last_error = exc
            time.sleep(2**attempt)
    assert last_error is not None
    raise last_error


def embed_texts(
    texts: list[str],
    *,
    client: OpenAI | None = None,
    show_progress: bool = False,
    concurrency: int | None = None,
) -> np.ndarray:
    """批量并发 embedding，返回 shape=(N, EMBED_DIM) 的 float32，已 L2 归一化。"""
    if not texts:
        return np.zeros((0, EMBED_DIM), dtype="float32")

    client = client or get_openai_client()
    prepared = [sanitize_embed_text(t) for t in texts]
    batches = [
        prepared[i : i + BATCH_SIZE] for i in range(0, len(prepared), BATCH_SIZE)
    ]
    total_batches = len(batches)
    workers = max(1, concurrency if concurrency is not None else EMBED_CONCURRENCY)
    # 单批或并发=1 时走串行，避免线程开销
    if total_batches == 1 or workers == 1:
        vectors: list[list[float]] = []
        for bi, batch in enumerate(batches, 1):
            if show_progress:
                print(f"  Embedding {bi}/{total_batches}", flush=True)
            vectors.extend(_embed_batch(client, batch))
        embeddings = np.array(vectors, dtype="float32")
        return _l2_normalize(embeddings)

    if show_progress:
        print(
            f"  Embedding {total_batches} 批并发（workers={workers}, batch_size={BATCH_SIZE}）",
            flush=True,
        )

    # 按批号回填，保证与 texts 顺序一致
    results: list[list[list[float]] | None] = [None] * total_batches
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {
            pool.submit(_embed_batch, client, batch): bi
            for bi, batch in enumerate(batches)
        }
        for future in as_completed(future_map):
            bi = future_map[future]
            results[bi] = future.result()
            done += 1
            if show_progress and (done % 20 == 0 or done == total_batches):
                print(f"  Embedding 完成 {done}/{total_batches}", flush=True)

    all_embeddings: list[list[float]] = []
    for part in results:
        assert part is not None
        all_embeddings.extend(part)

    embeddings = np.array(all_embeddings, dtype="float32")
    return _l2_normalize(embeddings)


def embed_query(query: str, *, client: OpenAI | None = None) -> np.ndarray:
    """单条查询向量，shape=(1, EMBED_DIM)。"""
    return embed_texts([query], client=client, show_progress=False, concurrency=1)
