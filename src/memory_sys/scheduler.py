"""HEARTBEAT：解析 HEARTBEAT.md，用 APScheduler 注册 cron，mtime 热重载。"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime
from typing import Awaitable, Callable

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from src.common.config import HEARTBEAT_RELOAD_SEC, get_chat_client
from src.memory_sys.flush import MemoryFlusher
from src.memory_sys.heartbeat import HeartbeatParser
from src.memory_sys.loader import MemoryLoader
from src.memory_sys.session_db import SessionDB

logger = logging.getLogger(__name__)
BroadcastFn = Callable[[str, dict], Awaitable[None]]


class HeartbeatScheduler:
    def __init__(
        self,
        parser: HeartbeatParser | None = None,
        flusher: MemoryFlusher | None = None,
    ) -> None:
        self._scheduler = AsyncIOScheduler()
        self._parser = parser or HeartbeatParser()
        self._flusher = flusher
        self._broadcast: BroadcastFn | None = None
        self._last_mtime: float = 0.0

    def start(self, broadcast: BroadcastFn) -> None:
        self._broadcast = broadcast
        self.reload_tasks()
        self._scheduler.add_job(
            self._check_reload,
            trigger="interval",
            seconds=HEARTBEAT_RELOAD_SEC,
            id="_heartbeat_watcher",
            replace_existing=True,
        )
        self._scheduler.start()
        logger.info("HeartbeatScheduler 已启动")

    def stop(self) -> None:
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)

    def reload_tasks(self) -> int:
        for job in self._scheduler.get_jobs():
            if not str(job.id).startswith("_"):
                job.remove()
        tasks = self._parser.load_tasks(enabled_only=True)
        for task in tasks:
            self._register_task(task)
        if self._parser.path.exists():
            self._last_mtime = self._parser.path.stat().st_mtime
        logger.info("已加载 %s 个 HEARTBEAT 任务", len(tasks))
        return len(tasks)

    def _register_task(self, task: dict) -> None:
        parts = str(task.get("trigger") or "").split()
        if len(parts) != 5:
            logger.error("trigger 格式错误：%s", task.get("trigger"))
            return
        minute, hour, day, month, day_of_week = parts
        trigger = CronTrigger(
            minute=minute,
            hour=hour,
            day=day,
            month=month,
            day_of_week=day_of_week,
        )
        self._scheduler.add_job(
            self._execute_task,
            trigger=trigger,
            args=[task],
            id=task["name"],
            replace_existing=True,
            misfire_grace_time=300,
        )

    async def _check_reload(self) -> None:
        if not self._parser.path.exists() or self._broadcast is None:
            return
        mtime = self._parser.path.stat().st_mtime
        if mtime <= self._last_mtime:
            return
        count = self.reload_tasks()
        await self._broadcast(
            "heartbeat_reloaded",
            {"message": "HEARTBEAT.md 已更新，任务重新加载", "task_count": count},
        )

    async def _execute_task(self, task: dict) -> None:
        if self._broadcast is None:
            return
        name = task["name"]
        action = task.get("action", "send_message")
        await self._broadcast(
            "heartbeat_start",
            {
                "task_name": name,
                "action": action,
                "description": task.get("description", ""),
                "fired_at": datetime.now().isoformat(),
            },
        )
        try:
            if action == "send_message":
                await self._action_send_message(task)
            elif action == "summarize_sessions":
                await self._action_summarize_sessions(task)
            elif action == "compact_memory":
                await self._action_compact_memory(task)
            elif action == "user_profile_refresh":
                await self._action_user_profile_refresh(task)
            else:
                logger.warning("未知 HEARTBEAT action：%s", action)
        except Exception as exc:  # noqa: BLE001 — 调度失败必须广播，不能打断事件循环
            logger.error("任务 %s 执行失败：%s", name, exc, exc_info=True)
            await self._broadcast("heartbeat_error", {"task_name": name, "error": str(exc)})

    async def _emit_message(self, task: dict, message: str) -> None:
        if self._broadcast is None:
            return
        await self._broadcast(
            "heartbeat_message",
            {
                "task_name": task["name"],
                "description": task.get("description", ""),
                "message": message,
                "fired_at": datetime.now().strftime("%H:%M"),
            },
        )

    async def _action_send_message(self, task: dict) -> None:
        loader = MemoryLoader()
        user_md = loader.get_user_md_path().read_text(encoding="utf-8")
        prompt = task.get("prompt") or "根据用户画像，生成一条简短友好的主动问候或提醒。"
        system = f"你是有记忆的金融分析助手，现在主动向用户发一条消息。\n用户画像：\n{user_md}"

        def _call() -> str:
            client, model = get_chat_client()
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.8,
            )
            return (resp.choices[0].message.content or "").strip()

        message = await asyncio.to_thread(_call)
        await self._emit_message(task, message)

    async def _action_summarize_sessions(self, task: dict) -> None:
        db = SessionDB()
        messages = db.get_today_messages()
        if not messages:
            await self._emit_message(task, "今日暂无对话记录，跳过汇总。")
            return
        conversation = "\n".join(
            f"{'用户' if item['role'] == 'user' else '助手'}：{item['content']}"
            for item in messages[-40:]
        )

        def _call() -> str:
            client, model = get_chat_client()
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": f"请用3~5句话总结以下对话的主要内容和要点：\n\n{conversation}",
                    }
                ],
                temperature=0.3,
            )
            return (resp.choices[0].message.content or "").strip()

        summary = await asyncio.to_thread(_call)
        flusher = self._flusher or MemoryFlusher()
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        flusher.append_entries(
            [
                {
                    "category": "event",
                    "title": f"对话汇总 {datetime.now().strftime('%Y-%m-%d')}",
                    "content": summary,
                    "date": now,
                    "id": f"summary_{datetime.now().strftime('%Y%m%d%H%M')}",
                }
            ]
        )
        await self._emit_message(task, f"对话已汇总并写入记忆：\n{summary}")

    async def _action_compact_memory(self, task: dict) -> None:
        flusher = self._flusher or MemoryFlusher()
        count = flusher.loader.get_memory_entry_count()
        if count < 5:
            await self._emit_message(task, f"当前记忆条目仅 {count} 条，无需压缩。")
            return
        before, after = await asyncio.to_thread(flusher.compact_memory)
        await self._emit_message(task, f"Memory Compaction 完成：{before} → {after} 条")

    async def _action_user_profile_refresh(self, task: dict) -> None:
        loader = MemoryLoader()
        memory_md = loader.get_memory_md_path().read_text(encoding="utf-8")
        user_md = loader.get_user_md_path().read_text(encoding="utf-8")

        def _call() -> str:
            client, model = get_chat_client()
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "根据以下长期记忆，更新用户画像文件 USER.md。"
                            "只输出更新后的完整 USER.md，不要其他说明。\n\n"
                            f"当前 USER.md：\n{user_md}\n\n长期记忆：\n{memory_md}"
                        ),
                    }
                ],
                temperature=0.1,
            )
            return (resp.choices[0].message.content or "").strip()

        raw = await asyncio.to_thread(_call)
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", raw.strip())
        cleaned = re.sub(r"\n?```$", "", cleaned.strip()).strip()
        loader.get_user_md_path().write_text(cleaned, encoding="utf-8")
        await self._emit_message(task, "USER.md 已根据最新记忆刷新。")
