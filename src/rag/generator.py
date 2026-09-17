"""根据检索摘录生成带引用答案；低相关则拒答。"""

from __future__ import annotations

from typing import Any

from openai import OpenAI

from src.common.config import (
    LLM_TIMEOUT_SEC,
    OPENAI_MODEL,
    SCORE_THRESHOLD,
    get_openai_client,
)

SYSTEM_PROMPT = """你是金融文档助手。仅依据给定摘录回答；不知则说不知。
每条结论后标注来源 [source:p.page]，其中 source 为文档名、page 为页码；无页码时写 [source:p.?]。
不要编造摘录中没有的数字或事实。"""

REFUSAL = "根据提供的摘录无法回答此问题，资料中没有足够依据。"

# LLM 拒答常用措辞；命中则清空 citations，避免「拒答却列来源」
_REFUSAL_MARKERS = (
    "无法回答",
    "没有足够依据",
    "资料中没有",
    "未能找到",
    "不足以支撑",
    "不知",
)


def _looks_like_refusal(text: str) -> bool:
    if text == REFUSAL:
        return True
    return any(marker in text for marker in _REFUSAL_MARKERS)


def _cite_label(hit: dict[str, Any]) -> str:
    source = hit.get("source") or "unknown"
    page = hit.get("page_num")
    page_str = str(page) if page is not None else "?"
    return f"[{source}:p.{page_str}]"


def _build_context(hits: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    parts: list[str] = []
    cites: list[dict[str, Any]] = []
    for i, hit in enumerate(hits, 1):
        label = _cite_label(hit)
        text = hit.get("text") or ""
        section = hit.get("section_path") or ""
        header = f"摘录{i} {label}"
        if section:
            header += f" · {section}"
        parts.append(f"{header}\n{text}")
        cites.append(
            {
                "index": i,
                "source": hit.get("source", ""),
                "page_num": hit.get("page_num"),
                "chunk_id": hit.get("chunk_id", ""),
                "label": label,
                "score": hit.get("score"),
            }
        )
    return "\n\n---\n\n".join(parts), cites


def answer(
    question: str,
    hits: list[dict[str, Any]],
    *,
    client: OpenAI | None = None,
    score_threshold: float = SCORE_THRESHOLD,
) -> tuple[str, list[dict[str, Any]]]:
    """返回 (answer_text, citations)。低分或空 hits 时拒答且不调 LLM。"""
    if not hits:
        return REFUSAL, []

    top_score = float(hits[0].get("score") or 0.0)
    if top_score < score_threshold:
        return REFUSAL, []

    context, cites = _build_context(hits)
    user_msg = (
        f"【摘录】\n{context}\n\n"
        f"【问题】\n{question}\n\n"
        "请仅依据摘录作答，并在结论后标注来源 [source:p.page]。"
    )

    client = client or get_openai_client()
    # 覆盖客户端默认 timeout，生成允许更长
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.1,
        timeout=LLM_TIMEOUT_SEC,
    )
    text = (resp.choices[0].message.content or "").strip()
    # 阈值未触发时 LLM 仍可能拒答；此时不应附带检索来源
    if _looks_like_refusal(text):
        return text, []
    return text, cites
