---
name: weather-query
description: >
  查询城市当前天气与未来 3 天预报。当用户问天气、气温、下雨、湿度、
  风速，或说「北京天气怎么样」「宁德明天热不热」时使用。
  先 load_skill，再调用 query_weather。
version: 0.1.0
---

# Weather Query — 天气查询

## 何时使用

- 用户问某地现在天气 / 气温 / 是否下雨
- 用户要未来几天简要预报

## 推荐流程

1. `load_skill("weather-query")`（Harness 会解锁 `query_weather`）
2. 调用 `query_weather(city="城市名")`
3. 用工具返回的报告组织回答，不要编造气象数据

## 注意

- 城市名尽量用常见写法；歧义地名可加「市」（如「宁德市」）
- 数据来自 Open-Meteo，需能访问外网；失败时据错误信息如实告知

## 更多

- API 与返回字段说明见 [references/notes.md](references/notes.md)
