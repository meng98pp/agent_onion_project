import os
import sys

# `python src/common/config.py` 会把本目录插到 sys.path[0]，
# 导致本地 types.py 遮蔽标准库 types（GenericAlias 循环导入）。
_this_dir = os.path.abspath(os.path.dirname(__file__))
if sys.path and os.path.normcase(os.path.abspath(sys.path[0])) == os.path.normcase(
    _this_dir
):
    sys.path.pop(0)

from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

ROOT = Path(__file__).resolve().parents[2]


def missing_required_keys() -> list[str]:
    required = ["DASHSCOPE_API_KEY"]
    return [k for k in required if not os.getenv(k)]


if __name__ == "__main__":
    missing = missing_required_keys()
    print("OK" if not missing else f"MISSING: {missing}")
