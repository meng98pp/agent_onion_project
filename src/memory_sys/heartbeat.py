"""解析 HEARTBEAT.md，并用正则初筛 + LLM 写入/取消定时任务。"""

from __future__ import annotations

import json
import logging
import re
from datetime import date
from pathlib import Path

from src.common.config import MEMORY_DIR, get_chat_client

logger = logging.getLogger(__name__)

SCHEDULE_PATTERNS = [
    re.compile(r"每天|每日"),
    re.compile(r"每周|每星期"),
    re.compile(r"每月"),
    re.compile(r"每\s*\d+\s*(分钟|小时|天|周)"),
    re.compile(r"定期|定时"),
    re.compile(r"提醒(我|一下)"),
    re.compile(r"帮我.*(记得|记住|定时|提醒)"),
    re.compile(r"自动.*(帮|给|整理|汇总|总结|提醒)"),
    re.compile(r"(早上|晚上|每天).*(提醒|告诉|汇报|说一下)"),
    re.compile(r"设置.*(提醒|任务|闹钟)"),
]
CANCEL_PATTERNS = [
    re.compile(r"取消.*(提醒|任务|定时|通知)"),
    re.compile(r"停止.*(提醒|任务|定时|发消息)"),
    re.compile(r"关闭.*(提醒|任务|定时)"),
    re.compile(r"删除.*(任务|提醒|定时)"),
    re.compile(r"禁用.*(任务|提醒)"),
    re.compile(r"不(要|用).*(提醒|任务|通知|发消息|发送)"),
    re.compile(r"别.*(提醒|发|通知|打扰)"),
    re.compile(r"(停|关)掉.*(提醒|任务|定时)"),
]
SUPPORTED_ACTIONS = {
    "send_message",
    "summarize_sessions",
    "compact_memory",
    "user_profile_refresh",
}
_INTENT_PROMPT = """\
用户说："{message}"

判断这是否是一个定时任务请求。若是，返回 JSON：
{{
  "is_schedule": true,
  "name": "snake_case任务名",
  "trigger": "cron五字段",
  "action": "send_message / summarize_sessions / compact_memory / user_profile_refresh",
  "description": "一句话描述",
  "prompt": "仅 send_message 时填写"
}}
若不是，返回：{{"is_schedule": false}}
只返回 JSON。"""


def _parse_json_object(text: str) -> dict | None:
    cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", text.strip())
    cleaned = re.sub(r"\n?```$", "", cleaned.strip())
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if not match:
        return None
    try:
        parsed = json.loads(match.group())
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


class HeartbeatParser:
    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else MEMORY_DIR / "HEARTBEAT.md"

    def load_tasks(self, *, enabled_only: bool = True) -> list[dict]:
        if not self.path.exists():
            return []
        content = self.path.read_text(encoding="utf-8")
        start = content.find("<!-- TASKS_START -->")
        end = content.find("<!-- TASKS_END -->")
        if start == -1 or end == -1:
            return []
        body = content[start + len("<!-- TASKS_START -->") : end].strip()
        tasks: list[dict] = []
        for block in re.split(r"(?=### TASK:)", body):
            block = block.strip()
            if not block.startswith("### TASK:"):
                continue
            task = self._parse_task_block(block)
            if not task:
                continue
            if enabled_only and not task.get("enabled", True):
                continue
            tasks.append(task)
        return tasks

    def _parse_task_block(self, block: str) -> dict | None:
        lines = block.splitlines()
        task = {"name": lines[0].replace("### TASK:", "").strip()}
        for line in lines[1:]:
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, _, val = line.partition(":")
            task[key.strip()] = val.strip()
        if "trigger" not in task or "action" not in task:
            return None
        task["enabled"] = str(task.get("enabled", "true")).lower() == "true"
        return task

    def may_contain_schedule_intent(self, message: str) -> bool:
        return any(pattern.search(message) for pattern in SCHEDULE_PATTERNS)

    def may_contain_cancel_intent(self, message: str) -> bool:
        return any(pattern.search(message) for pattern in CANCEL_PATTERNS)

    def analyze_and_write(self, message: str) -> dict | None:
        try:
            client, model = get_chat_client()
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": _INTENT_PROMPT.format(message=message)}],
                temperature=0,
            )
            data = _parse_json_object(resp.choices[0].message.content or "")
            if not data or not data.get("is_schedule"):
                return None
            if data.get("action") not in SUPPORTED_ACTIONS:
                logger.warning("不支持的 HEARTBEAT action：%s", data.get("action"))
                return None
            self._append_task(data)
            return data
        except Exception as exc:  # noqa: BLE001 — 意图分析失败时跳过，不打断对话
            logger.error("调度意图分析失败：%s", exc)
            return None

    def analyze_and_cancel(self, message: str) -> str | None:
        enabled = self.load_tasks(enabled_only=True)
        if not enabled:
            return None
        task_list = "\n".join(
            f"- {item['name']}：{item.get('description', item.get('action', ''))}"
            for item in enabled
        )
        prompt = (
            f'用户说："{message}"\n\n当前已启用的定时任务：\n{task_list}\n\n'
            '判断用户想停止哪个任务。能对应则返回 {"cancel": true, "name": "任务名"}；'
            '否则 {"cancel": false}。只返回 JSON。'
        )
        try:
            client, model = get_chat_client()
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            data = _parse_json_object(resp.choices[0].message.content or "")
            if not data or not data.get("cancel"):
                return None
            name = str(data.get("name") or "").strip()
            if not name:
                return None
            self._disable_task(name)
            return name
        except Exception as exc:  # noqa: BLE001 — 取消意图失败时跳过，不打断对话
            logger.error("取消意图分析失败：%s", exc)
            return None

    def _append_task(self, task: dict) -> None:
        content = self.path.read_text(encoding="utf-8")
        lines = [
            f"### TASK: {task['name']}",
            f"trigger: {task['trigger']}",
            "enabled: true",
            f"action: {task['action']}",
            f"description: {task.get('description', '')}",
        ]
        if task.get("prompt"):
            lines.append(f"prompt: {task['prompt']}")
        lines.append(f"added: {date.today().isoformat()}")
        block = "\n".join(lines) + "\n"
        self.path.write_text(
            content.replace("<!-- TASKS_END -->", block + "\n<!-- TASKS_END -->"),
            encoding="utf-8",
        )

    def _disable_task(self, task_name: str) -> None:
        content = self.path.read_text(encoding="utf-8")
        pattern = re.compile(
            r"(### TASK:\s*" + re.escape(task_name) + r".*?)(enabled:\s*true)",
            re.DOTALL,
        )
        new_content, count = pattern.subn(r"\1enabled: false", content, count=1)
        if count == 0:
            logger.warning("未找到任务 %s，无法禁用", task_name)
            return
        self.path.write_text(new_content, encoding="utf-8")
