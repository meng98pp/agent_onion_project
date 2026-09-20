"""Layer 3：七配置 Markdown → base system prompt。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from src.common.config import MEMORY_DIR, MEMORY_RECENT_N


@dataclass
class MemoryLayer:
    name: str
    source_file: str
    content: str
    char_count: int = 0

    def __post_init__(self) -> None:
        self.char_count = len(self.content)


@dataclass
class SystemPromptResult:
    system_prompt: str
    layers: list[MemoryLayer]
    total_chars: int = 0

    def __post_init__(self) -> None:
        self.total_chars = sum(layer.char_count for layer in self.layers)


class MemoryLoader:
    def __init__(self, memory_dir: Path | None = None) -> None:
        self.memory_dir = Path(memory_dir) if memory_dir is not None else MEMORY_DIR

    def _read_md(self, filename: str) -> str:
        path = self.memory_dir / filename
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8").strip()

    def extract_memory_entries(self, memory_md: str, limit: int) -> str:
        start = memory_md.find("<!-- MEMORY_ENTRIES_START -->")
        end = memory_md.find("<!-- MEMORY_ENTRIES_END -->")
        if start == -1 or end == -1:
            return ""
        body = memory_md[start + len("<!-- MEMORY_ENTRIES_START -->") : end].strip()
        if not body:
            return ""
        entries = [part.strip() for part in re.split(r"(?=### \[)", body) if part.strip()]
        recent = entries[-limit:] if len(entries) > limit else entries
        return "\n\n".join(recent)

    def _read_recent_day_logs(self, days: int = 2) -> tuple[str, list[str]]:
        today = date.today()
        parts: list[str] = []
        sources: list[str] = []
        for offset in range(days):
            day = today - timedelta(days=offset)
            path = self.memory_dir / f"{day.isoformat()}.md"
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8").strip()
            if not text:
                continue
            parts.append(f"### {day.isoformat()}\n{text}")
            sources.append(path.name)
        return ("\n\n".join(parts)).strip(), sources

    def build_system_prompt(self, recent_memory_limit: int | None = None) -> SystemPromptResult:
        limit = MEMORY_RECENT_N if recent_memory_limit is None else recent_memory_limit
        layers: list[MemoryLayer] = []
        parts: list[str] = []

        soul = self._read_md("SOUL.md")
        identity = self._read_md("IDENTITY.md")
        persona = "\n\n".join(p for p in (soul, identity) if p)
        if persona:
            layers.append(MemoryLayer("persona", "SOUL.md, IDENTITY.md", persona))
            parts.append(persona)

        agents = self._read_md("AGENTS.md")
        if agents:
            layers.append(MemoryLayer("agents_manual", "AGENTS.md", agents))
            parts.append(agents)

        user = self._read_md("USER.md")
        if user:
            section = f"## 关于当前用户\n{user}"
            layers.append(MemoryLayer("user_profile", "USER.md", section))
            parts.append(section)

        log_body, log_sources = self._read_recent_day_logs(days=2)
        if log_body:
            section = f"## 近期日志（今天 + 昨天）\n\n{log_body}"
            layers.append(MemoryLayer("daily_log", ", ".join(log_sources), section))
            parts.append(section)

        memory_md = self._read_md("MEMORY.md")
        memory_entries = self.extract_memory_entries(memory_md, limit)
        if memory_entries:
            section = f"## 长期记忆（最近 {limit} 条）\n\n{memory_entries}"
            layers.append(MemoryLayer("long_term_memory", "MEMORY.md", section))
            parts.append(section)

        return SystemPromptResult(system_prompt="\n\n---\n\n".join(parts), layers=layers)

    def get_memory_entry_count(self) -> int:
        return len(re.findall(r"^### \[", self._read_md("MEMORY.md"), re.MULTILINE))

    def get_user_md_path(self) -> Path:
        return self.memory_dir / "USER.md"

    def get_memory_md_path(self) -> Path:
        return self.memory_dir / "MEMORY.md"

    def layer_status(self, recent_memory_limit: int | None = None) -> list[dict]:
        """始终列出各层槽位（空文件也报 0 字符），方便 /layers 对照四层模型。"""
        limit = MEMORY_RECENT_N if recent_memory_limit is None else recent_memory_limit
        soul = self._read_md("SOUL.md")
        identity = self._read_md("IDENTITY.md")
        log_body, log_sources = self._read_recent_day_logs(days=2)
        memory_md = self._read_md("MEMORY.md")
        memory_entries = self.extract_memory_entries(memory_md, limit)
        return [
            {
                "id": "persona",
                "layer": 3,
                "label": "SOUL.md / IDENTITY.md（人设）",
                "source": "SOUL.md, IDENTITY.md",
                "chars": len(soul) + (2 if soul and identity else 0) + len(identity),
            },
            {
                "id": "agents_manual",
                "layer": 3,
                "label": "AGENTS.md（行为规范）",
                "source": "AGENTS.md",
                "chars": len(self._read_md("AGENTS.md")),
            },
            {
                "id": "user_profile",
                "layer": 3,
                "label": "USER.md（用户画像）",
                "source": "USER.md",
                "chars": len(self._read_md("USER.md")),
            },
            {
                "id": "daily_log",
                "layer": 2,
                "label": "每日日志（今天 + 昨天）",
                "source": ", ".join(log_sources) or "（无）",
                "chars": len(log_body),
            },
            {
                "id": "long_term_memory",
                "layer": 3,
                "label": f"MEMORY.md（最近 {limit} 条）",
                "source": "MEMORY.md",
                "chars": len(memory_entries),
            },
        ]


def attach_semantic_memories(system_prompt: str, results: list[dict]) -> str:
    if not results:
        return system_prompt
    snippets = [
        f"- [{item.get('category', '')}] {item.get('title', '')}: {str(item.get('content') or '')[:100]}"
        for item in results
    ]
    return system_prompt + "\n\n## 语义检索到的相关记忆\n" + "\n".join(snippets)


def load_base_prompt(recent_memory_limit: int | None = None) -> str:
    """验收入口：组装不含语义检索、不含会话历史的 base prompt。"""
    return MemoryLoader().build_system_prompt(recent_memory_limit).system_prompt


def format_layers_report(
    items: list[dict],
    *,
    session_messages: int = 0,
    faiss_total: int = 0,
    fts_total: int = 0,
    fts_available: bool = True,
    total_chars: int = 0,
) -> str:
    """把四层快照打成可读文本，供 /layers 打印。"""
    by_id = {item["id"]: item for item in items}
    lines = [
        "四层记忆加载情况",
        "────────────────────────────────",
        f"Layer 1  工作记忆（当次 Context 合计）  [{total_chars} 字符]",
        f"Layer 2  短期 · 日记（今+昨）           [{by_id.get('daily_log', {}).get('chars', 0)} 字符]",
        f"         短期 · 本会话 SQLite           [{session_messages} 条]",
        f"Layer 3  SOUL.md / IDENTITY.md          [{by_id.get('persona', {}).get('chars', 0)} 字符]",
        f"         AGENTS.md                      [{by_id.get('agents_manual', {}).get('chars', 0)} 字符]",
        f"         USER.md                        [{by_id.get('user_profile', {}).get('chars', 0)} 字符]",
        f"         MEMORY.md 近端条目             [{by_id.get('long_term_memory', {}).get('chars', 0)} 字符]",
        f"Layer 4  语义 · FAISS + FTS5            [faiss={faiss_total}  fts={fts_total}"
        + ("" if fts_available else "  FTS不可用")
        + "]",
        "────────────────────────────────",
    ]
    return "\n".join(lines)
