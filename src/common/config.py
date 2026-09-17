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

CHUNKS_DIR = ROOT / "data" / "chunks"
VECTORSTORE_DIR = ROOT / "vectorstore"
FAISS_INDEX_PATH = VECTORSTORE_DIR / "faiss_index.bin"
FAISS_META_PATH = VECTORSTORE_DIR / "faiss_meta.json"

OPENAI_BASE_URL = os.getenv(
    "OPENAI_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "qwen-plus")

EMBED_DIM = 1024
BATCH_SIZE = 10
# 同时发出的 embedding 批次数；过大易触发限流，可按额度调
EMBED_CONCURRENCY = int(os.getenv("EMBED_CONCURRENCY", "8"))
# DashScope text-embedding-v3：单条输入长度须在 [1, 8192]（接口按此校验）
EMBED_MAX_CHARS = 8192
SCORE_THRESHOLD = 0.25
DEFAULT_STRATEGY = "semantic"
LLM_TIMEOUT_SEC = 60.0
EMBED_TIMEOUT_SEC = 60.0


def missing_required_keys() -> list[str]:
    required = ["DASHSCOPE_API_KEY"]
    return [k for k in required if not os.getenv(k)]


def get_openai_client():
    """OpenAI 兼容客户端（DashScope）；缺 Key 时立刻报错。"""
    from openai import OpenAI

    missing = missing_required_keys()
    if missing:
        raise EnvironmentError(f"缺少环境变量: {', '.join(missing)}")
    return OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=OPENAI_BASE_URL,
        timeout=EMBED_TIMEOUT_SEC,
    )


if __name__ == "__main__":
    missing = missing_required_keys()
    print("OK" if not missing else f"MISSING: {missing}")
