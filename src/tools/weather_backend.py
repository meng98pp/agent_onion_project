"""
weather_backend — 天气查询业务后端（FC / MCP / CLI 共享）

纯 HTTP 业务逻辑；Open-Meteo 免费 API，无需 Key。
对外统一 ToolResult；MCP 可拆 geocode + fetch 做链式调用。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import httpx

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.common.types import ToolResult, fail, ok  # noqa: E402

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

WEATHER_CODE_MAP = {
    0: "晴天",
    1: "大致晴朗",
    2: "局部多云",
    3: "阴天",
    45: "雾",
    48: "冻雾",
    51: "小毛毛雨",
    53: "中毛毛雨",
    55: "大毛毛雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    80: "小阵雨",
    81: "中阵雨",
    82: "大阵雨",
    95: "雷暴",
    96: "雷暴伴小冰雹",
    99: "雷暴伴大冰雹",
}


def _resolve_location(city: str) -> dict[str, Any] | str:
    """成功返回 location dict；失败返回错误字符串。"""
    name = (city or "").strip()
    if not name:
        return "城市名不能为空"

    try:
        with httpx.Client(timeout=10.0) as client:

            def _geocode(q: str) -> list[dict]:
                resp = client.get(
                    GEOCODING_URL,
                    params={"name": q, "count": 10, "language": "zh", "format": "json"},
                )
                resp.raise_for_status()
                return resp.json().get("results") or []

            results = _geocode(name)
            # 裸「宁德」易命中西藏低级 PPL；无市/县/区后缀时用「市」重查
            is_low_admin = (
                all(
                    str(r.get("feature_code", "")).startswith("PPL")
                    and not str(r.get("feature_code", "")).startswith("PPLA")
                    for r in results
                )
                if results
                else True
            )
            has_suffix = any(name.endswith(s) for s in ("市", "县", "区", "镇"))
            if is_low_admin and not has_suffix:
                retry = _geocode(name + "市")
                if retry:
                    results = retry

            if not results:
                return f"未找到城市 '{name}'，请尝试其他写法（如加『市』）"

            def _rank(r: dict) -> tuple:
                fc = str(r.get("feature_code", ""))
                admin = 1 if fc.startswith("PPLA") or fc.startswith("ADM") else 0
                pop = r.get("population") or 0
                return (admin, pop)

            loc = max(results, key=_rank)
            city_name = loc.get("name", name)
            country = loc.get("country", "")
            admin1 = loc.get("admin1", "")
            return {
                "latitude": float(loc["latitude"]),
                "longitude": float(loc["longitude"]),
                "location_str": f"{country} {admin1} {city_name}".strip(),
            }
    except httpx.HTTPError as e:
        return f"地理编码请求失败：{e}"
    except Exception as e:
        return f"地理编码失败：{e}"


def geocode_city(city: str) -> ToolResult:
    """城市名 → 经纬度（MCP 链式第一步）。"""
    loc = _resolve_location(city)
    if isinstance(loc, str):
        return fail(loc)
    return ok(loc)


def fetch_weather(latitude: float, longitude: float) -> ToolResult:
    """经纬度 → 当前天气 + 未来 3 天预报。"""
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(
                WEATHER_URL,
                params={
                    "latitude": latitude,
                    "longitude": longitude,
                    "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
                    "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code",
                    "timezone": "Asia/Shanghai",
                    "forecast_days": 3,
                },
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as e:
        return fail(f"天气数据获取失败：{e}")
    except Exception as e:
        return fail(str(e))

    cur = data["current"]
    daily = data["daily"]
    weather_desc = WEATHER_CODE_MAP.get(cur["weather_code"], f"代码{cur['weather_code']}")
    forecast = []
    for i in range(min(3, len(daily.get("time") or []))):
        forecast.append(
            {
                "date": daily["time"][i],
                "desc": WEATHER_CODE_MAP.get(daily["weather_code"][i], ""),
                "temp_max": daily["temperature_2m_max"][i],
                "temp_min": daily["temperature_2m_min"][i],
                "precipitation_mm": daily["precipitation_sum"][i],
            }
        )

    return ok(
        {
            "latitude": latitude,
            "longitude": longitude,
            "current": {
                "desc": weather_desc,
                "temperature_c": cur["temperature_2m"],
                "humidity_pct": cur["relative_humidity_2m"],
                "wind_kmh": cur["wind_speed_10m"],
            },
            "forecast": forecast,
        }
    )


def query_weather(city: str) -> ToolResult:
    """一步到位：地理编码 + 天气（FC / CLI 用）。"""
    loc = _resolve_location(city)
    if isinstance(loc, str):
        return fail(loc)

    weather = fetch_weather(loc["latitude"], loc["longitude"])
    if not weather.get("ok"):
        return weather

    data = dict(weather["data"] or {})
    data["location_str"] = loc["location_str"]
    return ok(data)


def tool_result_text(result: ToolResult) -> str:
    if not result.get("ok"):
        return f"[错误] {result.get('error') or '未知错误'}"

    data = result.get("data") or {}
    if not isinstance(data, dict):
        return str(data)

    # 仅地理编码
    if "location_str" in data and "current" not in data and "latitude" in data:
        return (
            f"地理编码结果：{data['location_str']}\n"
            f"latitude={data['latitude']}\n"
            f"longitude={data['longitude']}\n"
            "（请用上述坐标调用 fetch_weather）"
        )

    loc = data.get("location_str") or (
        f"坐标 {data.get('latitude')}°N, {data.get('longitude')}°E"
    )
    cur = data.get("current") or {}
    lines = [
        f"【{loc}】天气报告",
        "",
        f"当前天气：{cur.get('desc', '')}",
        f"  温度：{cur.get('temperature_c')}°C",
        f"  相对湿度：{cur.get('humidity_pct')}%",
        f"  风速：{cur.get('wind_kmh')} km/h",
        "",
        "未来3天预报：",
    ]
    for day in data.get("forecast") or []:
        lines.append(
            f"  {day.get('date')}：{day.get('desc')}，"
            f"{day.get('temp_max')}°C / {day.get('temp_min')}°C，"
            f"降水 {day.get('precipitation_mm')} mm"
        )
    return "\n".join(lines)


# 兼容旧名
get_weather = query_weather


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True)
    args = parser.parse_args()
    r = query_weather(args.city)
    print(json.dumps(r, ensure_ascii=False, indent=2))
    print("---")
    print(tool_result_text(r))
