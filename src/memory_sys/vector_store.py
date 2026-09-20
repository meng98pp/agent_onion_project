"""Layer 4：记忆专用 FAISS（与年报 vectorstore 隔离）。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.common.config import EMBED_DIM, MEMORY_INDEX_DIR
from src.rag.embedder import embed_texts


class VectorStore:
    def __init__(self, index_dir: Path | None = None) -> None:
        self.index_dir = Path(index_dir) if index_dir is not None else MEMORY_INDEX_DIR
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.index_dir / "memory.faiss"
        self.meta_path = self.index_dir / "memory_meta.json"
        self.index = None
        self.metadata: list[dict] = []
        self._load()

    def _load(self) -> None:
        if not (self.index_path.exists() and self.meta_path.exists()):
            return
        import faiss

        self.index = faiss.read_index(str(self.index_path))
        self.metadata = json.loads(self.meta_path.read_text(encoding="utf-8"))

    def _save(self) -> None:
        import faiss

        if self.index is not None:
            faiss.write_index(self.index, str(self.index_path))
        self.meta_path.write_text(
            json.dumps(self.metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def add_entries(self, entries: list[dict]) -> int:
        import faiss

        if not entries:
            return 0
        texts = [str(item.get("content") or item.get("title") or ".") for item in entries]
        vectors = np.ascontiguousarray(embed_texts(texts), dtype="float32")
        if self.index is None:
            self.index = faiss.IndexFlatIP(int(vectors.shape[1] or EMBED_DIM))
        self.index.add(vectors)
        self.metadata.extend(entries)
        self._save()
        return len(entries)

    def rebuild_from_entries(self, entries: list[dict]) -> None:
        self.index = None
        self.metadata = []
        if self.index_path.exists():
            self.index_path.unlink()
        if self.meta_path.exists():
            self.meta_path.unlink()
        if entries:
            self.add_entries(entries)
        else:
            self._save()

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        if self.index is None or self.index.ntotal == 0:
            return []
        q_vec = np.ascontiguousarray(embed_texts([query]), dtype="float32")
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(q_vec, k)
        results: list[dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            item = dict(self.metadata[idx])
            item["score"] = float(score)
            results.append(item)
        return results

    @property
    def total_entries(self) -> int:
        return int(self.index.ntotal) if self.index is not None else 0
