"""把 ToolResult 转成 role=tool 回填文本。"""

from __future__ import annotations

import json
from typing import Any, Callable

from src.common.types import ToolResult


def result_for_llm(
    result: ToolResult,
    text_fn: Callable[[ToolResult], str] | None = None,
) -> str:
    """优先可读文本；失败时回退 JSON。"""
    if text_fn is not None:
        try:
            return text_fn(result)
        except Exception:
            pass
    return json.dumps(result, ensure_ascii=False)


def dispatch_tool(
    name: str,
    args: dict[str, Any],
    table: dict[str, Callable[..., ToolResult]],
    text_fns: dict[str, Callable[[ToolResult], str]] | None = None,
) -> str:
    fn = table.get(name)
    if fn is None:
        return f"未知工具：{name}"
    try:
        result = fn(**args)
    except TypeError as e:
        return f"参数错误：{e}"
    except Exception as e:
        return f"工具执行失败：{e}"
    text_fn = (text_fns or {}).get(name)
    return result_for_llm(result, text_fn)
