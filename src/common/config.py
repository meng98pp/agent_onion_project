import os
import sys

# FAISS(MKL) 与 PyTorch/sentence-transformers 常各带一份 OpenMP，Windows 上会触发 OMP Error #15。
# 须在加载 faiss/torch 之前设置；仅开发兼容，非官方长期方案。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

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

# V3：混合检索 / RRF / Rerank / 评估
RRF_K = 60
RETRIEVE_TOP_N = int(os.getenv("RETRIEVE_TOP_N", "10"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "3"))
MODELS_DIR = ROOT / "models"
# 本地 CrossEncoder：默认 models/bge-reranker-base（可用 RERANK_MODEL 覆盖目录名或绝对路径）
_RERANK_MODEL_ENV = os.getenv("RERANK_MODEL", "bge-small-zh-v1.5")
_rerank_path = Path(_RERANK_MODEL_ENV)
RERANK_MODEL_PATH = (
    _rerank_path if _rerank_path.is_absolute() else MODELS_DIR / _RERANK_MODEL_ENV
)
EVALUATION_DIR = ROOT / "evaluation"
EVAL_GOLD_PATH = EVALUATION_DIR / "gold_qa.json"
EVAL_RESULTS_DIR = EVALUATION_DIR / "results"


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


# V4：聊天 LLM（工具调用）；密钥仍只从本模块读取
CHAT_PROVIDERS = {
    "dashscope": {
        "env_key": "DASHSCOPE_API_KEY",
        "base_url": OPENAI_BASE_URL,
        "model": OPENAI_MODEL,
    },
    "deepseek": {
        "env_key": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
    },
}


def get_chat_client(provider: str = "dashscope"):
    """返回 (OpenAI 兼容 client, model_name)。"""
    from openai import OpenAI

    if provider not in CHAT_PROVIDERS:
        raise ValueError(f"未知 provider: {provider!r}，可选 {list(CHAT_PROVIDERS)}")
    cfg = CHAT_PROVIDERS[provider]
    api_key = os.getenv(cfg["env_key"], "")
    if not api_key:
        raise EnvironmentError(f"缺少环境变量: {cfg['env_key']}")
    client = OpenAI(api_key=api_key, base_url=cfg["base_url"], timeout=LLM_TIMEOUT_SEC)
    return client, cfg["model"]


if __name__ == "__main__":
    missing = missing_required_keys()
    print("OK" if not missing else f"MISSING: {missing}")
