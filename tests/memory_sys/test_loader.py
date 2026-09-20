"""验收：七配置能组装出非空 base prompt；缺文件不崩；MEMORY 近 N 条截取。"""

from __future__ import annotations

from pathlib import Path

from src.memory_sys.loader import MemoryLoader, load_base_prompt

ROOT = Path(__file__).resolve().parents[2]
MEMORY_DIR = ROOT / "memory"


def test_load_base_prompt_nonempty() -> None:
    prompt = load_base_prompt()
    assert len(prompt) > 0
    assert "FinAgent" in prompt or "金融" in prompt


def test_repo_memory_files_exist() -> None:
    for name in ("SOUL.md", "IDENTITY.md", "AGENTS.md", "USER.md", "HEARTBEAT.md", "MEMORY.md"):
        assert (MEMORY_DIR / name).exists(), name


def test_missing_files_do_not_crash(tmp_path: Path) -> None:
    loader = MemoryLoader(tmp_path)
    result = loader.build_system_prompt()
    assert result.system_prompt == ""
    assert result.layers == []
    assert loader.get_memory_entry_count() == 0


def test_assembly_order(tmp_path: Path) -> None:
    (tmp_path / "SOUL.md").write_text("SOUL_MARK", encoding="utf-8")
    (tmp_path / "IDENTITY.md").write_text("IDENTITY_MARK", encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("AGENTS_MARK", encoding="utf-8")
    (tmp_path / "USER.md").write_text("USER_MARK", encoding="utf-8")
    text = MemoryLoader(tmp_path).build_system_prompt().system_prompt
    assert text.find("SOUL_MARK") < text.find("IDENTITY_MARK") < text.find("AGENTS_MARK") < text.find("USER_MARK")


def test_memory_recent_n_keeps_latest(tmp_path: Path) -> None:
    memory_md = """# MEMORY.md

<!-- MEMORY_ENTRIES_START -->
### [fact] 第一条
记录时间：2026-01-01 00:00

旧条目

### [preference] 第二条
记录时间：2026-01-02 00:00

中间条目

### [event] 第三条
记录时间：2026-01-03 00:00

新条目
<!-- MEMORY_ENTRIES_END -->
"""
    (tmp_path / "MEMORY.md").write_text(memory_md, encoding="utf-8")
    loader = MemoryLoader(tmp_path)
    excerpt = loader.extract_memory_entries(memory_md, 2)
    assert "第一条" not in excerpt
    assert "第二条" in excerpt
    assert "第三条" in excerpt
    assert loader.get_memory_entry_count() == 3
