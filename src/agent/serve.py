"""FastAPI：V5 ReAct 轨迹 + V6 记忆对话 / Flush / HEARTBEAT。"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import src.common.config  # noqa: F401,E402
from src.common.config import (  # noqa: E402
    FLUSH_MESSAGE_THRESHOLD,
    HEARTBEAT_ENABLED,
    MAX_STEPS,
    OPENAI_MODEL,
    STATIC_DIR,
    get_chat_client,
)
from src.memory_sys.flush import MemoryFlusher  # noqa: E402
from src.memory_sys.fts_store import FTSStore  # noqa: E402
from src.memory_sys.heartbeat import HeartbeatParser  # noqa: E402
from src.memory_sys.loader import (  # noqa: E402
    MemoryLoader,
    attach_semantic_memories,
    format_layers_report,
    load_base_prompt,
)
from src.memory_sys.retrieval import HybridRetriever  # noqa: E402
from src.memory_sys.scheduler import HeartbeatScheduler  # noqa: E402
from src.memory_sys.session_db import SessionDB  # noqa: E402
from src.memory_sys.vector_store import VectorStore  # noqa: E402

logger = logging.getLogger(__name__)

db: SessionDB | None = None
loader: MemoryLoader | None = None
vs: VectorStore | None = None
fts: FTSStore | None = None
retriever: HybridRetriever | None = None
flusher: MemoryFlusher | None = None
hb_parser: HeartbeatParser | None = None
hb_scheduler: HeartbeatScheduler | None = None
current_session_id: int | None = None
_stream_listeners: list[asyncio.Queue] = []


async def broadcast(event_type: str, data: dict) -> None:
    """把 HEARTBEAT / 自动 Flush 事件推给所有 /stream 连接。"""
    payload = _sse({"type": event_type, **data})
    for queue in list(_stream_listeners):
        try:
            queue.put_nowait(payload)
        except asyncio.QueueFull:
            logger.warning("HEARTBEAT 广播队列已满，丢弃 %s", event_type)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """启动时初始化记忆组件与会话，关闭时停调度器并结束当前会话。"""
    global db, loader, vs, fts, retriever, flusher, hb_parser, hb_scheduler, current_session_id
    db = SessionDB()
    loader = MemoryLoader()
    vs = VectorStore()
    fts = FTSStore()
    retriever = HybridRetriever(vs, fts)
    flusher = MemoryFlusher(vs=vs, fts=fts)
    hb_parser = HeartbeatParser()
    current_session_id = db.new_session()
    if HEARTBEAT_ENABLED:
        hb_scheduler = HeartbeatScheduler(parser=hb_parser, flusher=flusher)
        hb_scheduler.start(broadcast)
    yield
    if hb_scheduler:
        hb_scheduler.stop()
    if current_session_id:
        db.close_session(current_session_id)


app = FastAPI(title="V6 Memory + ReAct Financial Agent", lifespan=lifespan)


class QueryRequest(BaseModel):
    """V5 ReAct 提问。"""

    question: str
    max_steps: int = MAX_STEPS


class ChatRequest(BaseModel):
    """V6 记忆对话。session_id 为空则用当前会话。"""

    message: str
    session_id: int | None = None


class FlushRequest(BaseModel):
    """手动 Flush。session_id 为空则用当前会话。"""

    session_id: int | None = None


def _sse(data: dict) -> str:
    """把 dict 编成 SSE data 行。"""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _extra_system() -> str:
    """读取七配置 Markdown，拼进 ReAct 的 system prompt。"""
    try:
        return load_base_prompt()
    except OSError as exc:
        logger.warning("加载记忆 prompt 失败：%s", exc)
        return ""


async def _stream_react(question: str, max_steps: int, mode: str):
    """在线程里跑 ReAct，把每步轨迹用 SSE 推给前端。"""
    if mode == "manual":
        from src.agent.react_manual import run as react_run
    else:
        from src.agent.react_fc import run as react_run

    extra = _extra_system()
    queue: asyncio.Queue = asyncio.Queue()
    sentinel = object()

    def _worker() -> None:
        """同步执行 ReAct，结果放入队列。"""
        try:
            for step_data in react_run(question, max_steps=max_steps, extra_system=extra or None):
                queue.put_nowait(step_data)
        except Exception as exc:  # noqa: BLE001 — 推给 SSE，避免线程内崩溃
            queue.put_nowait({"step": 0, "type": "error", "observation": f"循环异常：{exc}"})
        finally:
            queue.put_nowait(sentinel)

    yield _sse({"type": "start", "question": question, "mode": mode})
    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, _worker)
    while True:
        step_data = await queue.get()
        if step_data is sentinel:
            break
        yield _sse(step_data)
    yield _sse({"type": "done"})


@app.post("/query/manual")
async def query_manual(req: QueryRequest):
    """手写 Prompt 解析版 ReAct。"""
    return StreamingResponse(
        _stream_react(req.question, req.max_steps, "manual"),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/query/fc")
async def query_fc(req: QueryRequest):
    """原生 Function Calling 版 ReAct。"""
    return StreamingResponse(
        _stream_react(req.question, req.max_steps, "fc"),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/chat")
async def chat(req: ChatRequest):
    """记忆对话：组装四层 Context → 流式 LLM → 写入 SQLite。"""
    assert db and loader and retriever and flusher
    sid = req.session_id or current_session_id

    async def stream():
        """推送 memory_load / 检索 / token / done，必要时后台 Flush。"""
        prompt_result = loader.build_system_prompt()
        layers_info = [
            {"name": layer.name, "source": layer.source_file, "chars": layer.char_count}
            for layer in prompt_result.layers
        ]
        yield _sse({"type": "memory_load", "layers": layers_info, "total_chars": prompt_result.total_chars})

        semantic_results = retriever.search(req.message)
        yield _sse(
            {
                "type": "semantic_search",
                "query": req.message,
                "results": [
                    {
                        "category": item.get("category", ""),
                        "title": item.get("title", ""),
                        "content": str(item.get("content") or "")[:120],
                        "score": round(float(item.get("score") or 0), 3),
                        "source": item.get("source", ""),
                    }
                    for item in semantic_results
                ],
            }
        )

        system_prompt = attach_semantic_memories(prompt_result.system_prompt, semantic_results)
        history = db.get_session_messages(sid)
        history_for_api = [{"role": item["role"], "content": item["content"]} for item in history]
        yield _sse(
            {
                "type": "context_assembly",
                "system_chars": len(system_prompt),
                "history_turns": len(history_for_api),
                "layers_used": [row["name"] for row in layers_info]
                + (["faiss_semantic"] if semantic_results else []),
            }
        )

        api_messages = (
            [{"role": "system", "content": system_prompt}]
            + history_for_api
            + [{"role": "user", "content": req.message}]
        )
        queue: asyncio.Queue = asyncio.Queue()

        def _worker() -> None:
            """流式调用聊天模型，按 token 写入队列。"""
            try:
                client, model = get_chat_client()
                resp = client.chat.completions.create(
                    model=model, messages=api_messages, temperature=0.4, stream=True
                )
                chunks: list[str] = []
                for part in resp:
                    delta = part.choices[0].delta.content or ""
                    if delta:
                        chunks.append(delta)
                        queue.put_nowait(("token", delta))
                queue.put_nowait(("complete", "".join(chunks)))
            except Exception as exc:  # noqa: BLE001
                queue.put_nowait(("error", str(exc)))

        asyncio.get_running_loop().run_in_executor(None, _worker)
        full_response = ""
        while True:
            kind, payload = await queue.get()
            if kind == "token":
                yield _sse({"type": "token", "text": payload})
            elif kind == "complete":
                full_response = payload or "（模型返回空内容）"
                break
            else:
                full_response = f"（生成失败：{payload}）"
                yield _sse({"type": "error", "message": payload})
                break

        db.add_message(sid, "user", req.message)
        db.add_message(sid, "assistant", full_response)
        msg_count = db.get_message_count(sid)
        yield _sse(
            {
                "type": "done",
                "response": full_response,
                "session_id": sid,
                "message_count": msg_count,
                "auto_flush_threshold": FLUSH_MESSAGE_THRESHOLD,
            }
        )
        if msg_count >= FLUSH_MESSAGE_THRESHOLD:
            asyncio.create_task(_background_flush(sid))
        if hb_parser and (
            hb_parser.may_contain_schedule_intent(req.message)
            or hb_parser.may_contain_cancel_intent(req.message)
        ):
            asyncio.create_task(_check_schedule_intent(req.message))

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _background_flush(session_id: int) -> None:
    """消息数达标后异步 Flush，结果通过 /stream 广播。"""
    if not db or not flusher:
        return
    messages = db.get_session_messages(session_id)
    result = await asyncio.to_thread(flusher.flush, messages, session_id)
    if not result.error:
        db.mark_flushed(session_id)
    await broadcast(
        "auto_flush",
        {
            "session_id": session_id,
            "summary": result.summary(),
            "error": result.error,
            "message": result.error or result.summary(),
        },
    )


async def _check_schedule_intent(message: str) -> None:
    """正则初筛后让 LLM 判断新建/取消定时任务，并热重载调度器。"""
    if not hb_parser:
        return
    if hb_parser.may_contain_cancel_intent(message):
        name = await asyncio.to_thread(hb_parser.analyze_and_cancel, message)
        if name:
            if hb_scheduler:
                hb_scheduler.reload_tasks()
            await broadcast(
                "heartbeat_task_cancelled",
                {"task_name": name, "message": f"已停止定时任务：{name}"},
            )
            return
    if hb_parser.may_contain_schedule_intent(message):
        task = await asyncio.to_thread(hb_parser.analyze_and_write, message)
        if task:
            if hb_scheduler:
                hb_scheduler.reload_tasks()
            await broadcast(
                "heartbeat_task_added",
                {
                    "task_name": task["name"],
                    "trigger": task.get("trigger", ""),
                    "description": task.get("description", ""),
                    "message": f"已设置定时任务：{task.get('description', task['name'])}",
                },
            )


async def _flush_pass_events(sid: int):
    """对指定会话执行 Flush，按 Pass 产出事件 dict（不含 SSE 包装）。"""
    assert db and flusher and vs
    messages = db.get_session_messages(sid)
    yield {"type": "flush_start", "session_id": sid, "message_count": len(messages)}
    if not messages:
        yield {"type": "flush_done", "error": "会话为空"}
        return
    result = await asyncio.to_thread(flusher.flush, messages, sid)
    yield {
        "type": "flush_pass1",
        "user_updates": result.user_updates,
        "count": len(result.user_updates),
    }
    yield {
        "type": "flush_pass2",
        "new_entries": [
            {
                "category": item.get("category", ""),
                "title": item.get("title", ""),
                "content": str(item.get("content") or "")[:100],
            }
            for item in result.new_memory_entries
        ],
        "count": len(result.new_memory_entries),
        "daily_summary": result.daily_summary,
    }
    yield {
        "type": "flush_pass3",
        "vectorized": result.vectorized_count,
        "total_in_index": vs.total_entries,
    }
    if result.compacted:
        yield {
            "type": "flush_compaction",
            "before": result.compaction_before,
            "after": result.compaction_after,
        }
    if not result.error:
        db.mark_flushed(sid)
    yield {"type": "flush_done", "error": result.error, "summary": result.summary()}


@app.post("/flush")
async def flush_session(req: FlushRequest):
    """手动 Memory Flush：三写 USER / MEMORY / 日记，并推送各 Pass 进度。"""
    sid = req.session_id or current_session_id

    async def stream():
        """按 Pass1/2/3（及可选压缩）把 Flush 进度发给前端。"""
        async for event in _flush_pass_events(sid):
            yield _sse(event)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/layers")
async def get_layers():
    """打印四层记忆加载情况：工作/短期/长期/语义，含字符数与索引统计。"""
    assert db and loader and vs and fts
    items = loader.layer_status()
    session_messages = db.get_message_count(current_session_id) if current_session_id else 0
    total_chars = sum(item["chars"] for item in items)
    report = format_layers_report(
        items,
        session_messages=session_messages,
        faiss_total=vs.total_entries,
        fts_total=fts.total_entries,
        fts_available=fts.available,
        total_chars=total_chars,
    )
    return JSONResponse(
        {
            "items": items,
            "total_chars": total_chars,
            "session_id": current_session_id,
            "session_messages": session_messages,
            "faiss_total": vs.total_entries,
            "fts_total": fts.total_entries,
            "fts_available": fts.available,
            "report": report,
        }
    )


@app.get("/memories")
async def get_memories():
    """返回七配置 Markdown 正文与条目/索引统计，供侧栏展示。"""
    assert db and loader and vs and fts

    def read_md(name: str) -> str:
        """读 memory/ 下单个 md，缺失则空串。"""
        path = loader.memory_dir / name
        return path.read_text(encoding="utf-8") if path.exists() else ""

    return JSONResponse(
        {
            "user_md": read_md("USER.md"),
            "memory_md": read_md("MEMORY.md"),
            "soul_md": read_md("SOUL.md"),
            "identity_md": read_md("IDENTITY.md"),
            "agents_md": read_md("AGENTS.md"),
            "heartbeat_md": read_md("HEARTBEAT.md"),
            "entry_count": loader.get_memory_entry_count(),
            "faiss_total": vs.total_entries,
            "fts_total": fts.total_entries,
            "fts_available": fts.available,
            "recent_sessions": db.get_recent_sessions(5),
            "session_id": current_session_id,
        }
    )


@app.post("/session/new")
async def new_session():
    """关闭当前会话并开新会话。不对旧会话 Flush；USER/MEMORY 保持原样。"""
    global current_session_id
    assert db
    if current_session_id:
        db.close_session(current_session_id)
    current_session_id = db.new_session()
    return {"session_id": current_session_id}


@app.post("/session/exit")
async def exit_session(req: FlushRequest):
    """ /exit：先 Flush 当前会话，再关闭并开新窗口（页面可继续用）。"""
    global current_session_id
    assert db
    sid = req.session_id or current_session_id

    async def stream():
        """Flush 各 Pass 之后再发 session_exited。"""
        global current_session_id
        async for event in _flush_pass_events(sid):
            yield _sse(event)
        if sid:
            db.close_session(sid)
        current_session_id = db.new_session()
        yield _sse(
            {
                "type": "session_exited",
                "old_session_id": sid,
                "session_id": current_session_id,
            }
        )

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/stream")
async def stream_events():
    """持久 SSE：接收 HEARTBEAT 广播与自动 Flush 通知。"""
    queue: asyncio.Queue = asyncio.Queue(maxsize=50)
    _stream_listeners.append(queue)

    async def generate():
        """先推当前任务列表，再阻塞等待广播；超时发 keepalive。"""
        try:
            tasks = hb_parser.load_tasks() if hb_parser else []
            yield _sse(
                {
                    "type": "heartbeat_connected",
                    "task_count": len(tasks),
                    "tasks": [
                        {
                            "name": item["name"],
                            "trigger": item["trigger"],
                            "description": item.get("description", ""),
                        }
                        for item in tasks
                    ],
                }
            )
            while True:
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=20.0)
                    yield payload
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            if queue in _stream_listeners:
                _stream_listeners.remove(queue)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/health")
async def health():
    """探活：模型、会话、记忆条数、FTS / HEARTBEAT 是否可用。"""
    return {
        "status": "ok",
        "model": OPENAI_MODEL,
        "session_id": current_session_id,
        "memory_entries": loader.get_memory_entry_count() if loader else 0,
        "faiss_entries": vs.total_entries if vs else 0,
        "fts_available": fts.available if fts else False,
        "heartbeat": bool(hb_scheduler),
    }


@app.get("/")
async def root():
    """返回 Web UI（记忆对话 + ReAct 轨迹）。"""
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>static/index.html not found</h2>")
