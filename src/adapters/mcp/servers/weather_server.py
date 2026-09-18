"""
weather_server.py — 天气 MCP Server（stdio）

暴露 geocode_city / fetch_weather 供 Host 链式调用；业务在 src.tools.weather_backend。
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mcp.server.mcpserver import MCPServer  # noqa: E402

from src.tools.weather_backend import (  # noqa: E402
    fetch_weather as _fetch_weather,
    geocode_city as _geocode_city,
    tool_result_text as _weather_text,
)


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


mcp = MCPServer("weather-server")


@mcp.tool()
def geocode_city(city: str) -> str:
    """
    将城市名解析为经纬度。查天气前须先调用本工具，再把坐标传给 fetch_weather。

    Args:
        city: 城市中文名，如 '宁德'、'北京'。
    """
    return _weather_text(_geocode_city(city))


@mcp.tool()
def fetch_weather(latitude: float, longitude: float) -> str:
    """
    按经纬度查询当前天气及未来3天预报。坐标须来自 geocode_city。

    Args:
        latitude: 纬度，例如 26.67
        longitude: 经度，例如 119.52
    """
    return _weather_text(_fetch_weather(latitude, longitude))


if __name__ == "__main__":
    log("Weather MCP Server 启动中（stdio）...")
    mcp.run(transport="stdio")
