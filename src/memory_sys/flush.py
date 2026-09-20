"""Memory Flush：三写 USER / MEMORY / 日记，并更新 FAISS+FTS；超阈值压缩。"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

from src.common.config import (
    MEMORY_COMPACTION_KEEP_RECENT,
    MEMORY_COMPACTION_THRESHOLD,
    MEMORY_DIR,
    get_chat_client,
)
from src.memory_sys.flush_prompts import (
    COMPACTION_PROMPT,
    MEMORY_EXTRACT_PROMPT,
    SESSION_SUMMARY_PROMPT,
    USER_EXTRACT_PROMPT,
    USER_MD_UPDATE_PROMPT,
)
from src.memory_sys.fts_store import FTSStore
from src.memory_sys.loader import MemoryLoader
from src.memory_sys.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class FlushResult:
    session_id: int
    user_updates: list[str] = field(default_factory=list)
    new_memory_entries: list[dict] = field(default_factory=list)
    daily_summary: str = ""
    compacted: bool = False
    compaction_before: int = 0
    compaction_after: int = 0
    vectorized_count: int = 0
    error: str = ""

    def summary(self) -> str:
        lines = [f"=== Memory Flush 结果（会话 #{self.session_id}）==="]
        if self.error:
            lines.append(f"[错误] {self.error}")
            return "\n".join(lines)
        lines.append(f"USER.md 更新项：{len(self.user_updates)} 条")
        lines.extend(f"  · {item}" for item in self.user_updates)
        lines.append(f"新增长期记忆：{len(self.new_memory_entries)} 条")
        for entry in self.new_memory_entries:
            lines.append(f"  [{entry.get('category', '?')}] {entry.get('title', '')}")
        if self.daily_summary:
            preview = self.daily_summary[:120]
            suffix = "…" if len(self.daily_summary) > 120 else ""
            lines.append(f"每日日志摘要：{preview}{suffix}")
        if self.compacted:
            lines.append(f"Compaction：{self.compaction_before} → {self.compaction_after} 条")
        lines.append(f"向量化写入：{self.vectorized_count} 条")
        return "\n".join(lines)


def _chat(messages: list[dict], temperature: float = 0.1) -> str:
    client, model = get_chat_client()
    resp = client.chat.completions.create(
        model=model, messages=messages, temperature=temperature
    )
    return (resp.choices[0].message.content or "").strip()


def _strip_code_fence(text: str) -> str:
    cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", text.strip())
    return re.sub(r"\n?```$", "", cleaned.strip()).strip()


def _parse_json_array(text: str) -> list:
    cleaned = _strip_code_fence(text)
    match = re.search(r"\[[\s\S]*\]", cleaned)
    if not match:
        return []
    try:
        parsed = json.loads(match.group())
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


class MemoryFlusher:
    def __init__(
        self,
        memory_dir: Path | None = None,
        vs: VectorStore | None = None,
        fts: FTSStore | None = None,
        compaction_threshold: int | None = None,
    ) -> None:
        self.loader = MemoryLoader(memory_dir or MEMORY_DIR)
        self.vs = vs if vs is not None else VectorStore()
        self.fts = fts if fts is not None else FTSStore()
        self.compaction_threshold = (
            MEMORY_COMPACTION_THRESHOLD if compaction_threshold is None else compaction_threshold
        )

    def flush(self, messages: list[dict], session_id: int) -> FlushResult:
        result = FlushResult(session_id=session_id)
        conversation = self._format_conversation(messages)
        if not conversation.strip():
            result.error = "会话为空，跳过 Flush"
            return result
        try:
            result.user_updates = self._extract_and_update_user(conversation)
            new_entries = self._extract_memory_entries(conversation)
            if new_entries:
                self.append_entries(new_entries)
                result.new_memory_entries = new_entries
            result.daily_summary = self._summarize_session(conversation)
            self._append_to_daily_log(session_id, result.daily_summary, new_entries)
            if new_entries:
                result.vectorized_count = self.vs.add_entries(new_entries)
                self.fts.add_entries(new_entries)
            count = self.loader.get_memory_entry_count()
            if count >= self.compaction_threshold:
                before, after = self.compact_memory()
                result.compacted = True
                result.compaction_before = before
                result.compaction_after = after
        except Exception as exc:  # noqa: BLE001 — Flush 必须返回 error 字段给 SSE，不能中断会话
            result.error = str(exc)
            logger.error("Flush 失败: %s", exc, exc_info=True)
        return result

    def append_entries(self, entries: list[dict]) -> None:
        path = self.loader.get_memory_md_path()
        content = path.read_text(encoding="utf-8")
        end_marker = "<!-- MEMORY_ENTRIES_END -->"
        blocks = []
        for entry in entries:
            blocks.append(
                f"### [{entry.get('category', 'fact')}] {entry.get('title', '未命名')}\n"
                f"记录时间：{entry.get('date', '')}\n\n"
                f"{entry.get('content', '')}"
            )
        insertion = "\n\n".join(blocks) + "\n\n"
        path.write_text(content.replace(end_marker, insertion + end_marker), encoding="utf-8")

    def compact_memory(self) -> tuple[int, int]:
        path = self.loader.get_memory_md_path()
        content = path.read_text(encoding="utf-8")
        start_marker = "<!-- MEMORY_ENTRIES_START -->"
        end_marker = "<!-- MEMORY_ENTRIES_END -->"
        start = content.find(start_marker)
        end = content.find(end_marker)
        if start == -1 or end == -1:
            return 0, 0
        body = content[start + len(start_marker) : end].strip()
        entries_raw = [part.strip() for part in re.split(r"(?=### \[)", body) if part.strip()]
        total_before = len(entries_raw)
        if total_before <= MEMORY_COMPACTION_KEEP_RECENT:
            return total_before, total_before
        old_entries = entries_raw[:-MEMORY_COMPACTION_KEEP_RECENT]
        recent_entries = entries_raw[-MEMORY_COMPACTION_KEEP_RECENT:]
        compact_resp = _chat(
            [
                {
                    "role": "user",
                    "content": COMPACTION_PROMPT.format(
                        entries=json.dumps([{"text": item} for item in old_entries], ensure_ascii=False)
                    ),
                }
            ]
        )
        compacted_raw = _parse_json_array(compact_resp)
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        compacted_blocks = []
        for item in compacted_raw:
            if isinstance(item, dict) and "content" in item:
                compacted_blocks.append(
                    f"### [{item.get('category', 'fact')}] {item.get('title', '压缩摘要')}\n"
                    f"记录时间：{now}（压缩整理）\n\n{item['content']}"
                )
            else:
                compacted_blocks.append(
                    f"### [fact] 压缩摘要\n记录时间：{now}（压缩整理）\n\n{item}"
                )
        all_blocks = compacted_blocks + recent_entries
        new_body = "\n\n".join(all_blocks)
        new_content = (
            content[: start + len(start_marker)] + "\n" + new_body + "\n\n" + content[end:]
        )
        path.write_text(new_content, encoding="utf-8")
        all_entries = self.parse_memory_entries()
        self.vs.rebuild_from_entries(all_entries)
        self.fts.rebuild_from_entries(all_entries)
        return total_before, len(all_blocks)

    def parse_memory_entries(self) -> list[dict]:
        content = self.loader.get_memory_md_path().read_text(encoding="utf-8")
        start = content.find("<!-- MEMORY_ENTRIES_START -->")
        end = content.find("<!-- MEMORY_ENTRIES_END -->")
        if start == -1 or end == -1:
            return []
        blocks = re.split(r"(?=### \[)", content[start:end])
        entries: list[dict] = []
        for i, block in enumerate(blocks):
            text = block.strip()
            match = re.match(
                r"### \[(\w+)\]\s+(.+)\n记录时间：([^\n]+)\n\n([\s\S]+)", text
            )
            if not match:
                continue
            entries.append(
                {
                    "id": f"entry_{i}",
                    "category": match.group(1),
                    "title": match.group(2),
                    "date": match.group(3),
                    "content": f"{match.group(2)}: {match.group(4).strip()}",
                }
            )
        return entries

    @staticmethod
    def _format_conversation(messages: list[dict]) -> str:
        lines = []
        for item in messages:
            role = "用户" if item.get("role") == "user" else "助手"
            lines.append(f"{role}：{item.get('content', '')}")
        return "\n".join(lines)

    def _extract_and_update_user(self, conversation: str) -> list[str]:
        path = self.loader.get_user_md_path()
        current = path.read_text(encoding="utf-8")
        extract_resp = _chat(
            [{"role": "user", "content": USER_EXTRACT_PROMPT.format(conversation=conversation)}]
        )
        new_info = _parse_json_array(extract_resp)
        if not new_info:
            return []
        update_resp = _chat(
            [
                {
                    "role": "user",
                    "content": USER_MD_UPDATE_PROMPT.format(
                        current_user_md=current,
                        new_info=json.dumps(new_info, ensure_ascii=False, indent=2),
                    ),
                }
            ]
        )
        path.write_text(_strip_code_fence(update_resp), encoding="utf-8")
        return [f"{item.get('field', '?')}: {item.get('value', '')}" for item in new_info if isinstance(item, dict)]

    def _extract_memory_entries(self, conversation: str) -> list[dict]:
        extract_resp = _chat(
            [{"role": "user", "content": MEMORY_EXTRACT_PROMPT.format(conversation=conversation)}]
        )
        entries = [item for item in _parse_json_array(extract_resp) if isinstance(item, dict)]
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        stamp = now.replace(" ", "_").replace(":", "")
        for i, entry in enumerate(entries):
            entry.setdefault("date", now)
            entry.setdefault("id", f"entry_{stamp}_{i}")
        return entries

    def _summarize_session(self, conversation: str) -> str:
        resp = _chat(
            [{"role": "user", "content": SESSION_SUMMARY_PROMPT.format(conversation=conversation)}]
        )
        return _strip_code_fence(resp).strip() or "（本会话无有效摘要）"

    def _append_to_daily_log(
        self, session_id: int, summary: str, related_entries: list[dict] | None = None
    ) -> None:
        log_path = self.loader.memory_dir / f"{date.today().isoformat()}.md"
        now = datetime.now().strftime("%H:%M")
        lines = [f"\n## {now} 会话 #{session_id}\n", "### 会话摘要", summary.strip(), ""]
        if related_entries:
            lines.append("### 关联长期记忆（已写入 MEMORY.md）")
            for entry in related_entries:
                lines.append(f"- [{entry.get('category', 'fact')}] {entry.get('title', '')}")
            lines.append("")
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(lines))
