"""Skill 热插拔加载器：L0 索引 / L1 正文 / L2 参考文件；tools.py 延迟 import。"""

from __future__ import annotations

import importlib.util
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from src.common.config import ROOT, SKILLS_DIR

logger = logging.getLogger(__name__)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ToolHandler = Callable[..., Any]

_TOOL_NAME_RE = re.compile(r"""["']name["']\s*:\s*["']([A-Za-z0-9_-]+)["']""")


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler
    skill: str

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class SkillInfo:
    name: str
    description: str
    version: str
    dir: Path
    skill_md_path: Path
    body: str
    tool_names: list[str] = field(default_factory=list)
    tools: list[ToolSpec] = field(default_factory=list)
    tools_loaded: bool = False
    mtime: float = 0.0


def parse_frontmatter(raw: str) -> tuple[dict[str, str], str]:
    """解析 Cursor 风格 YAML frontmatter（最小实现，不依赖 PyYAML）。"""
    text = raw.lstrip("\ufeff")
    if not text.startswith("---"):
        return {}, text

    rest = text[3:]
    if rest.startswith("\r\n"):
        rest = rest[2:]
    elif rest.startswith("\n"):
        rest = rest[1:]

    end = re.search(r"\n---\s*\r?\n", rest)
    if not end:
        end2 = re.search(r"\n---\s*$", rest)
        if not end2:
            return {}, text
        fm_text = rest[: end2.start()]
        body = ""
    else:
        fm_text = rest[: end.start()]
        body = rest[end.end() :]

    meta: dict[str, str] = {}
    lines = fm_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip() or line.strip().startswith("#"):
            i += 1
            continue
        m = re.match(r"^([A-Za-z0-9_-]+)\s*:\s*(.*)$", line)
        if not m:
            i += 1
            continue
        key, val = m.group(1), m.group(2).strip()
        if val in (">", "|", ">-", "|-"):
            folded = val.startswith(">")
            block: list[str] = []
            i += 1
            while i < len(lines):
                nxt = lines[i]
                if nxt.startswith(" ") or nxt.startswith("\t") or nxt == "":
                    block.append(nxt[2:] if nxt.startswith("  ") else nxt.lstrip("\t"))
                    i += 1
                    continue
                break
            meta[key] = (
                " ".join(x.strip() for x in block if x.strip())
                if folded
                else "\n".join(block).strip()
            )
            continue
        if (val.startswith('"') and val.endswith('"')) or (
            val.startswith("'") and val.endswith("'")
        ):
            val = val[1:-1]
        meta[key] = val
        i += 1

    return meta, body


def _peek_tool_names(tools_path: Path) -> list[str]:
    """只从 tools.py 源码抽出 name，不 exec（L0 索引用）。"""
    if not tools_path.is_file():
        return []
    try:
        text = tools_path.read_text(encoding="utf-8")
    except OSError:
        return []
    names: list[str] = []
    seen: set[str] = set()
    for match in _TOOL_NAME_RE.finditer(text):
        name = match.group(1)
        if name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def _scan_mtime(skills_dir: Path) -> float:
    latest = 0.0
    if not skills_dir.is_dir():
        return latest
    for pattern in ("*/SKILL.md", "*/tools.py"):
        for path in skills_dir.glob(pattern):
            try:
                latest = max(latest, path.stat().st_mtime)
            except OSError:
                continue
    return latest


def _load_skill_tools(skill_name: str, skill_dir: Path) -> list[ToolSpec]:
    """动态加载 skills/<name>/tools.py 中的 TOOLS（仅在 load_skill 后调用）。"""
    tools_path = skill_dir / "tools.py"
    if not tools_path.is_file():
        return []

    module_name = f"_skill_tools_{skill_name.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, tools_path)
    if spec is None or spec.loader is None:
        logger.warning("无法加载 %s", tools_path)
        return []

    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        logger.error("执行 %s 失败: %s", tools_path, exc, exc_info=True)
        return []

    raw_tools = getattr(mod, "TOOLS", None)
    if not isinstance(raw_tools, list):
        return []

    specs: list[ToolSpec] = []
    for item in raw_tools:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        handler = item.get("handler")
        params = item.get("parameters")
        if not name or not callable(handler) or not isinstance(params, dict):
            logger.warning("技能 %s 的工具项无效（需 name/handler/parameters）", skill_name)
            continue
        specs.append(
            ToolSpec(
                name=name,
                description=str(item.get("description") or name),
                parameters=params,
                handler=handler,
                skill=skill_name,
            )
        )
    return specs


class SkillLoader:
    def __init__(self, skills_dir: Path | None = None) -> None:
        self.skills_dir = Path(skills_dir) if skills_dir else SKILLS_DIR
        self._skills: dict[str, SkillInfo] = {}
        self._scan_mtime: float = 0.0
        self.reload()

    def reload(self) -> int:
        """扫描 skills/*/SKILL.md：只解析 frontmatter + 工具名；不 import tools.py。"""
        self._skills.clear()
        if not self.skills_dir.is_dir():
            self.skills_dir.mkdir(parents=True, exist_ok=True)
            self._scan_mtime = 0.0
            return 0

        for skill_md in sorted(self.skills_dir.glob("*/SKILL.md")):
            try:
                raw = skill_md.read_text(encoding="utf-8")
                mtime = skill_md.stat().st_mtime
            except OSError:
                continue
            tools_path = skill_md.parent / "tools.py"
            try:
                if tools_path.is_file():
                    mtime = max(mtime, tools_path.stat().st_mtime)
            except OSError:
                pass
            meta, body = parse_frontmatter(raw)
            name = (meta.get("name") or skill_md.parent.name).strip()
            if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", name):
                name = skill_md.parent.name
            desc = (meta.get("description") or "").strip()
            if not desc:
                desc = f"Skill `{name}`（未填写 description）"
            skill_dir = skill_md.parent.resolve()
            info = SkillInfo(
                name=name,
                description=desc,
                version=(meta.get("version") or "").strip(),
                dir=skill_dir,
                skill_md_path=skill_md.resolve(),
                body=body.strip(),
                tool_names=_peek_tool_names(tools_path),
                mtime=mtime,
            )
            self._skills[name] = info
        self._scan_mtime = _scan_mtime(self.skills_dir)
        return len(self._skills)

    def maybe_reload_if_stale(self) -> bool:
        """mtime 变化则热加载。返回是否发生了 reload。"""
        current = _scan_mtime(self.skills_dir)
        if current != self._scan_mtime:
            self.reload()
            return True
        return False

    @property
    def skills(self) -> dict[str, SkillInfo]:
        return self._skills

    def list_l0(self) -> list[dict[str, Any]]:
        self.maybe_reload_if_stale()
        rows: list[dict[str, Any]] = []
        for skill in self._skills.values():
            row: dict[str, Any] = {
                "name": skill.name,
                "description": skill.description,
                "tools": list(skill.tool_names),
            }
            if skill.version:
                row["version"] = skill.version
            rows.append(row)
        return rows

    def build_l0_prompt_section(self) -> str:
        self.maybe_reload_if_stale()
        lines = [
            "## 可用技能（L0 目录 — 渐进式披露）",
            "此处**只有**技能名与简介；业务工具 Schema **默认不在** tools 列表中。",
            "流程：`load_skill(name)` → 获得说明书并解锁该技能工具 → 再调用；",
            "L2 细节用 `read_skill_resource`。元工具始终可用：`list_skills` / `load_skill` / `read_skill_resource`。",
            "",
        ]
        if not self._skills:
            lines.append("（当前 skills/ 下暂无技能，可新增 `skills/<name>/SKILL.md`[+tools.py] 后 list_skills 或 /reload。）")
            return "\n".join(lines)
        for skill in self._skills.values():
            ver = f" v{skill.version}" if skill.version else ""
            tool_hint = (
                f"（解锁工具：{', '.join(skill.tool_names)}）"
                if skill.tool_names
                else "（无业务工具，仅说明书）"
            )
            lines.append(f"- **{skill.name}**{ver}：{skill.description} {tool_hint}")
        return "\n".join(lines)

    def ensure_tools_loaded(self, name: str) -> list[ToolSpec]:
        """命中技能后才 import tools.py。"""
        skill = self._skills.get(name)
        if not skill:
            return []
        if not skill.tools_loaded:
            skill.tools = _load_skill_tools(skill.name, skill.dir)
            if skill.tools:
                skill.tool_names = [item.name for item in skill.tools]
            skill.tools_loaded = True
        return skill.tools

    def load_l1(self, name: str) -> str:
        self.maybe_reload_if_stale()
        skill = self._skills.get(name)
        if not skill:
            available = ", ".join(self._skills) or "(无)"
            return f"未找到技能 `{name}`。当前可用：{available}。可先调用 list_skills。"
        tools = self.ensure_tools_loaded(name)
        header = f"# Skill: {skill.name}\n"
        if skill.version:
            header += f"version: {skill.version}\n"
        header += f"description: {skill.description}\n"
        if tools:
            header += (
                "unlocked_tools: "
                + ", ".join(item.name for item in tools)
                + "\n（以上工具已在本轮 Function Calling 中解锁）\n"
            )
        header += "\n"
        return header + (skill.body or "（SKILL.md 正文为空）")

    def read_resource(self, skill_name: str, relative_path: str) -> str:
        self.maybe_reload_if_stale()
        skill = self._skills.get(skill_name)
        if not skill:
            return f"未找到技能 `{skill_name}`。"

        rel = (relative_path or "").strip().replace("\\", "/")
        if not rel or rel.startswith("/") or ".." in rel.split("/"):
            return f"非法路径：{relative_path!r}（禁止绝对路径与 ..）"

        root = skill.dir.resolve()
        target = (root / rel).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            return "路径越界：禁止读取技能目录之外的文件。"

        if not target.is_file():
            return f"文件不存在：{skill_name}/{rel}"

        try:
            data = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return f"无法以文本读取：{rel}（可能是二进制文件）"
        except OSError as exc:
            return f"读取失败：{exc}"

        if len(data) > 80_000:
            return data[:80_000] + "\n\n…（已截断，文件过大）"
        return data

    def iter_tool_specs(self, skill_names: set[str] | None = None) -> list[ToolSpec]:
        """收集已加载的业务工具。skill_names=None 表示全部技能（MCP 导出时会先 ensure）。"""
        specs: list[ToolSpec] = []
        seen: set[str] = set()
        for skill in self._skills.values():
            if skill_names is not None and skill.name not in skill_names:
                continue
            for item in skill.tools:
                if item.name in seen:
                    specs = [x for x in specs if x.name != item.name]
                seen.add(item.name)
                specs.append(item)
        return specs

    def skills_providing_tool(self, tool_name: str) -> list[str]:
        owners: list[str] = []
        for skill in self._skills.values():
            names = [item.name for item in skill.tools] if skill.tools_loaded else skill.tool_names
            if tool_name in names:
                owners.append(skill.name)
        return owners


_default_loader: SkillLoader | None = None


def get_skill_loader() -> SkillLoader:
    global _default_loader
    if _default_loader is None:
        _default_loader = SkillLoader()
    return _default_loader


def bind_skill_loader(loader: SkillLoader) -> SkillLoader:
    global _default_loader
    _default_loader = loader
    return loader


def load_skill_index(skills_dir: Path | None = None) -> list[dict[str, Any]]:
    """启动用 L0 索引：name / description / 工具名。不注入 SKILL 正文。"""
    if skills_dir is not None:
        return SkillLoader(skills_dir).list_l0()
    loader = get_skill_loader()
    loader.maybe_reload_if_stale()
    return loader.list_l0()
