"""入口：读 data/raw，parse → 三种分块 → 写 data/parsed 与 data/chunks。

本文件不写解析/分块算法，只编排。python src/ingest/run_chunk.py 可直接跑
（下面把仓库根目录插进 sys.path）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.common.config import ROOT
from src.ingest.chunking import STRATEGIES, chunks_to_dicts
from src.ingest.parse_docs import parse_file


def iter_raw_files(raw_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in raw_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".txt", ".pdf"}
    )


def dump_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def process_file(
    path: Path,
    chunks_dir: Path,
    parsed_dir: Path | None = None,
) -> list[Path]:
    def progress(page: int, total: int) -> None:
        # 年报页数多、find_tables 较慢，隔页打印以免看起来像卡住
        if page == 1 or page == total or page % 10 == 0:
            print(f"  {path.name}: page {page}/{total}", flush=True)

    blocks = parse_file(path, progress=progress)
    print(f"  {path.name}: {len(blocks)} blocks", flush=True)
    if parsed_dir is not None:
        parsed_dir.mkdir(parents=True, exist_ok=True)
        dump_json(parsed_dir / f"{path.stem}.json", blocks)

    written: list[Path] = []
    chunks_dir.mkdir(parents=True, exist_ok=True)
    for name, fn in STRATEGIES.items():
        chunks = chunks_to_dicts(fn(blocks))
        out_path = chunks_dir / f"{path.stem}_{name}.json"
        dump_json(out_path, chunks)
        written.append(out_path)
        print(f"  {path.name} {name}: {len(chunks)} chunks", flush=True)
    return written


def run_chunk(
    raw_dir: Path | None = None,
    chunks_dir: Path | None = None,
    parsed_dir: Path | None = None,
) -> list[Path]:
    raw_dir = raw_dir or (ROOT / "data" / "raw")
    chunks_dir = chunks_dir or (ROOT / "data" / "chunks")
    parsed_dir = parsed_dir or (ROOT / "data" / "parsed")
    raw_dir.mkdir(parents=True, exist_ok=True)

    files = iter_raw_files(raw_dir)
    if not files:
        print(f"data/raw 下没有 .txt/.pdf: {raw_dir}", flush=True)
        return []

    print(f"found {len(files)} file(s) in {raw_dir}", flush=True)
    written: list[Path] = []
    for path in files:
        try:
            written.extend(process_file(path, chunks_dir, parsed_dir))
            print(f"parsed {path.name}", flush=True)
        except Exception as exc:
            print(f"FAILED {path.name}: {exc}", flush=True)
    print("done", flush=True)
    return written


if __name__ == "__main__":
    run_chunk()
