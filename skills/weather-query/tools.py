"""weather-query 技能工具。包装 src.tools.weather_backend，不复制气象算法。"""

from __future__ import annotations

from src.tools.weather_backend import query_weather

TOOLS = [
    {
        "name": "query_weather",
        "description": (
            "查询指定城市的当前天气与未来 3 天预报（温度、湿度、风速、降水）。"
            "城市名支持中文，如「北京」「上海」「宁德市」。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名称，例如「北京」「宁德」",
                },
            },
            "required": ["city"],
        },
        "handler": query_weather,
    },
]
