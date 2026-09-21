# weather-query 备注（L2）

仅在调用 `read_skill_resource("weather-query", "references/notes.md")` 时进入上下文。

数据源：Open-Meteo（免费、无需 API Key）

| 步骤 | API |
|------|-----|
| 地名 → 经纬度 | `geocoding-api.open-meteo.com` |
| 天气 | `api.open-meteo.com/v1/forecast` |

工具 `query_weather(city)` 一步完成地理编码 + 预报。

依赖：`httpx`（项目 requirements 已包含）
