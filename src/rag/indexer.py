"""离线建索引：读 semantic chunks → Embedding → FAISS IndexFlatIP + 元数据。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.common.config import (
    CHUNKS_DIR,
    DEFAULT_STRATEGY,
    EMBED_DIM,
    EMBED_MAX_CHARS,
    FAISS_INDEX_PATH,
    FAISS_META_PATH,
    VECTORSTORE_DIR,
    get_openai_client,
)
from src.rag.embedder import embed_texts


def load_chunks(
    chunks_dir: Path | None = None,
    strategy: str = DEFAULT_STRATEGY,
) -> list[dict]:
    """合并 data/chunks/*_{strategy}.json。"""
    chunks_dir = chunks_dir or CHUNKS_DIR
    paths = sorted(chunks_dir.glob(f"*_{strategy}.json"))
    if not paths:
        raise FileNotFoundError(
            f"未找到分块文件: {chunks_dir}/*_{strategy}.json（请先跑 V1 run_chunk.py）"
        )

    chunks: list[dict] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"分块文件应为 JSON 数组: {path}")
        chunks.extend(payload)
    return chunks


def chunk_to_meta(chunk: dict) -> dict:
    """按契约字段提取元数据；缺省给安全默认值。"""
    return {
        "chunk_id": str(chunk.get("chunk_id", "")),
        "text": str(chunk.get("text", "")),
        "source": str(chunk.get("source", "")),
        "page_num": chunk.get("page_num"),
        "section_path": str(chunk.get("section_path", "")),
        "strategy": str(chunk.get("strategy", DEFAULT_STRATEGY)),
        "parent_id": chunk.get("parent_id"),
        "block_types": list(chunk.get("block_types") or []),
        "stock_code": str(chunk.get("stock_code", "")),
        "year": str(chunk.get("year", "")),
    }


def build_index(
    chunks: list[dict] | None = None,
    *,
    strategy: str = DEFAULT_STRATEGY,
    show_progress: bool = True,
):
    """构建 FAISS IndexFlatIP，返回 (index, meta_list)。"""
    import faiss

    if chunks is None:
        chunks = load_chunks(strategy=strategy)
    if not chunks:
        raise ValueError("没有可索引的 chunks")

    meta_list = [chunk_to_meta(c) for c in chunks]
    texts = [m["text"] for m in meta_list]
    empty_n = sum(1 for t in texts if not t.strip())
    over_n = sum(1 for t in texts if len(t.strip()) > EMBED_MAX_CHARS)
    if empty_n:
        print(f"警告: {empty_n} 条空文本将用占位符 '.' 做 embedding", flush=True)
    if over_n:
        print(
            f"警告: {over_n} 条超过 {EMBED_MAX_CHARS} 字，embedding 时截断"
            f"（元数据仍保留全文）",
            flush=True,
        )

    client = get_openai_client()
    print(f"开始 embedding {len(texts)} 条 chunks...", flush=True)
    embeddings = embed_texts(texts, client=client, show_progress=show_progress)

    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(np.ascontiguousarray(embeddings)) #把 embeddings 整理成 FAISS 能安全读取的c连续内存格式，按行存储。
    print(f"FAISS 索引完成: ntotal={index.ntotal}, dim={EMBED_DIM}", flush=True)
    return index, meta_list


def save_index(index, meta_list: list[dict], *, vectorstore_dir: Path | None = None) -> None:
    import faiss

    vectorstore_dir = vectorstore_dir or VECTORSTORE_DIR
    vectorstore_dir.mkdir(parents=True, exist_ok=True)
    index_path = vectorstore_dir / FAISS_INDEX_PATH.name
    meta_path = vectorstore_dir / FAISS_META_PATH.name

    faiss.write_index(index, str(index_path))
    meta_path.write_text(
        json.dumps(meta_list, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"已保存索引 → {index_path}", flush=True)
    print(f"已保存元数据 → {meta_path} ({len(meta_list)} 条)", flush=True)


def load_index(*, vectorstore_dir: Path | None = None):
    """加载 FAISS 索引与元数据，返回 (index, meta_list)。"""
    import faiss

    vectorstore_dir = vectorstore_dir or VECTORSTORE_DIR
    index_path = vectorstore_dir / FAISS_INDEX_PATH.name
    meta_path = vectorstore_dir / FAISS_META_PATH.name
    if not index_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"索引不存在，请先运行 python src/rag/indexer.py\n"
            f"  期望: {index_path}\n  期望: {meta_path}"
        )

    index = faiss.read_index(str(index_path))
    meta_list = json.loads(meta_path.read_text(encoding="utf-8"))
    if index.ntotal != len(meta_list):
        raise ValueError(
            f"索引条数与元数据不一致: index={index.ntotal} meta={len(meta_list)}"
        )
    return index, meta_list


def main() -> None:
    index, meta_list = build_index()
    save_index(index, meta_list)


if __name__ == "__main__":
    main()
