"""按后缀把文件交给 parse_txt / parse_pdf，对外只暴露 parse_file(path) -> list[block]。"""

from __future__ import annotations

from pathlib import Path

from src.ingest.parse_pdf import parse_pdf
from src.ingest.parse_txt import parse_txt

PARSERS = {
    ".txt": parse_txt,
    ".pdf": parse_pdf,
}


def parse_file(path: str | Path, progress=None) -> list[dict]:
    path = Path(path)
    suffix = path.suffix.lower()
    parser = PARSERS.get(suffix)
    if parser is None:
        raise ValueError(f"unsupported file type: {suffix or '(none)'}")
    if suffix == ".pdf":
        return parse_pdf(path, progress=progress)
    return parse_txt(path)
