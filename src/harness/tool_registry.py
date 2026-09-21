"""工具注册表：元工具常驻 + Skill 业务工具热插拔 + Schema 渐进披露。"""

from __future__ import annotations

import json
from typing import Any, Callable

from src.harness.skill_loader import get_skill_loader
from src.tools import TEXT_FNS

ToolHandler = Callable[..., Any]

META_TOOL_NAMES = frozenset({"list_skills", "load_skill", "read_skill_resource"})


def _tool_list_skills() -> str:
    loader = get_skill_loader()
    loader.reload()
    rebuild_handlers()
    rows = loader.list_l0()
    if not rows:
        return "当前没有已发现的技能。请在 skills/<name>/SKILL.md 添加后重试。"
    lines = [f"共 {len(rows)} 个技能（L0）："]
    for row in rows:
        ver = f" v{row['version']}" if row.get("version") else ""
        tools = ", ".join(row.get("tools") or []) or "无"
        lines.append(f"- {row['name']}{ver}: {row['description']} [tools: {tools}]")
    return "\n".join(lines)


def _tool_load_skill(name: str) -> str:
    loader = get_skill_loader()
    return loader.load_l1(name.strip())


def _tool_read_skill_resource(skill: str, path: str) -> str:
    loader = get_skill_loader()
    return loader.read_resource(skill.strip(), path.strip())


_META_HANDLERS: dict[str, ToolHandler] = {
    "list_skills": _tool_list_skills,
    "load_skill": _tool_load_skill,
    "read_skill_resource": _tool_read_skill_resource,
}

_META_TOOLS_SCHEMA: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_skills",
            "description": "列出已发现技能（L0）及其可解锁工具名；会热重载 skills/ 目录。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_skill",
            "description": (
                "加载技能说明书（L1），并在本轮对话中解锁该技能 tools.py 里的业务工具 Schema。"
                "必须先调用本工具，才能使用对应业务工具。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "技能名，如 stock-analyst、weather-query",
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_skill_resource",
            "description": "读取技能目录内参考文件（L2）。禁止路径穿越。",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill": {"type": "string", "description": "技能名"},
                    "path": {
                        "type": "string",
                        "description": "相对路径，如 references/tool-notes.md",
                    },
                },
                "required": ["skill", "path"],
            },
        },
    },
]

TOOLS_MAP: dict[str, ToolHandler] = dict(_META_HANDLERS)


def rebuild_handlers(*, load_all_skill_tools: bool = False) -> dict[str, ToolHandler]:
    """按当前 SkillLoader 重建 TOOLS_MAP。默认只登记已 ensure 过的业务工具。"""
    loader = get_skill_loader()
    if load_all_skill_tools:
        for name in list(loader.skills):
            loader.ensure_tools_loaded(name)
    TOOLS_MAP.clear()
    TOOLS_MAP.update(_META_HANDLERS)
    for spec in loader.iter_tool_specs(skill_names=None):
        TOOLS_MAP[spec.name] = spec.handler
    return TOOLS_MAP


def get_tools_schema(
    *,
    activated_skills: set[str] | None = None,
    include_all_skill_tools: bool = False,
) -> list[dict[str, Any]]:
    """
    生成 OpenAI tools= 列表。

    - 默认 / activated_skills=空集：仅元工具（渐进披露起点）
    - activated_skills={'stock-analyst', ...}：元工具 + 这些技能的业务 Schema
    - include_all_skill_tools=True：元工具 + 全部技能工具（供 MCP 导出）
    """
    loader = get_skill_loader()
    loader.maybe_reload_if_stale()

    if include_all_skill_tools:
        rebuild_handlers(load_all_skill_tools=True)
        specs = loader.iter_tool_specs(skill_names=None)
    elif activated_skills:
        for name in activated_skills:
            loader.ensure_tools_loaded(name)
        rebuild_handlers()
        specs = loader.iter_tool_specs(skill_names=activated_skills)
    else:
        rebuild_handlers()
        specs = []

    schema = list(_META_TOOLS_SCHEMA)
    seen: set[str] = set(META_TOOL_NAMES)
    for spec in specs:
        if spec.name in seen:
            continue
        seen.add(spec.name)
        schema.append(spec.to_openai_schema())
    return schema


def build_tools_prompt_section() -> str:
    get_skill_loader().maybe_reload_if_stale()
    rebuild_handlers()
    return get_skill_loader().build_l0_prompt_section()


def _to_observation(name: str, result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict) and "ok" in result:
        text_fn = TEXT_FNS.get(name)
        if text_fn is not None:
            try:
                return text_fn(result)
            except Exception:
                pass
        if name == "query_weather":
            try:
                from src.tools.weather_backend import tool_result_text as weather_text

                return weather_text(result)
            except Exception:
                pass
        return json.dumps(result, ensure_ascii=False)
    return str(result)


def call_tool(
    name: str,
    arguments: dict[str, Any] | str | None = None,
    *,
    activated_skills: set[str] | None = None,
) -> str:
    """执行工具。传入 activated_skills 时，未解锁的业务工具会被拒绝。"""
    rebuild_handlers()

    if name not in META_TOOL_NAMES and activated_skills is not None:
        loader = get_skill_loader()
        owners = loader.skills_providing_tool(name)
        if not owners:
            return f"未知业务工具: {name}"
        if not (set(owners) & activated_skills):
            hint = " 或 ".join(f"load_skill('{owner}')" for owner in owners)
            return f"工具 `{name}` 属于技能 {owners}，尚未解锁。请先 {hint}。"

    if name not in TOOLS_MAP:
        if name not in META_TOOL_NAMES and activated_skills:
            loader = get_skill_loader()
            for skill_name in activated_skills:
                loader.ensure_tools_loaded(skill_name)
            rebuild_handlers()
        if name not in TOOLS_MAP:
            return f"未知工具: {name}，当前可用：{', '.join(sorted(TOOLS_MAP))}"

    if arguments is None:
        arguments = {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError as exc:
            return f"工具参数 JSON 解析失败: {exc}，原始参数: {arguments}"
    if not isinstance(arguments, dict):
        return f"工具参数必须是对象，收到: {type(arguments).__name__}"

    if name == "rag_search" and "top_k" in arguments:
        try:
            arguments["top_k"] = int(arguments["top_k"])
        except (TypeError, ValueError):
            arguments.pop("top_k", None)

    try:
        result = TOOLS_MAP[name](**arguments)
    except TypeError as exc:
        return f"工具 {name} 参数错误: {exc}，收到参数: {arguments}"
    except Exception as exc:
        return f"工具 {name} 执行出错: {exc}"
    return _to_observation(name, result)


def is_load_skill_success(result: str, name: str) -> bool:
    """判断 load_skill 返回是否表示成功（用于 ReAct 解锁）。"""
    if not result:
        return False
    if result.startswith("未找到技能"):
        return False
    return f"# Skill: {name}" in result or result.startswith("# Skill:")
