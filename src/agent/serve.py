"""FastAPI + SSE：把 ReAct 每步推给 Web UI。"""

from __future__ import annotations

import asyncio
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import src.common.config  # noqa: F401,E402
from src.common.config import MAX_STEPS, OPENAI_MODEL, STATIC_DIR  # noqa: E402


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield


app = FastAPI(title="V5 ReAct Financial Agent", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str
    max_steps: int = MAX_STEPS


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _stream_react(question: str, max_steps: int, mode: str):
    if mode == "manual":
        from src.agent.react_manual import run as react_run
    else:
        from src.agent.react_fc import run as react_run

    queue: asyncio.Queue = asyncio.Queue()
    sentinel = object()

    def _worker() -> None:
        try:
            for step_data in react_run(question, max_steps=max_steps):
                queue.put_nowait(step_data)
        except Exception as e:
            queue.put_nowait(
                {"step": 0, "type": "error", "observation": f"循环异常：{e}"}
            )
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
    return StreamingResponse(
        _stream_react(req.question, req.max_steps, "manual"),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/query/fc")
async def query_fc(req: QueryRequest):
    return StreamingResponse(
        _stream_react(req.question, req.max_steps, "fc"),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model": OPENAI_MODEL}


@app.get("/")
async def root():
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>static/index.html not found</h2>")
