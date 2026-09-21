# 开发规范（审查以本文件为准）

粒度：**小功能 = 函数**；**大功能 = 一个 `.py`**；**领域 = 一个目录**。

## 目录（随版本叠加，不要预建空壳）

```text
src/common/      # 配置、类型、日志
src/ingest/      # V1 解析分块
src/rag/         # V2–V3 检索生成评估
src/tools/       # 纯业务后端（无 MCP/CLI/Prompt）
src/adapters/    # function_call / mcp / cli
src/agent/       # ReAct 与 HTTP
src/memory_sys/  # 记忆 Python（仓库根 memory/ 只放 Markdown）
src/harness/     # Skill 加载与注册
src/evolve/      # 评估与自进化
tests/<同名领域>/
```

## 依赖方向

`common` ← 所有人。`ingest` 不进运行时 Agent。`adapters` 不写算法。`tools` 不写协议。`evolve` 不直接改检索实现。

## 鲁棒

- 对外返回 `{ok, data, error}`，禁止裸 `except:`
- LLM / HTTP / 子进程必须 timeout
- ReAct：`MAX_STEPS` + 重复动作熔断
- 计算器用 AST 白名单；CLI 命令白名单
- 配置只从 `src/common/config.py` 读

## 命名

| 对象 | 规则 |
|------|------|
| 文件/目录 | snake_case |
| 函数 | 动词开头 snake_case |
| 类 | PascalCase |
| 常量 | UPPER_SNAKE |
| 测试 | `tests/<domain>/test_<模块>.py` |
| commit | `feat(vN):` / `fix(vN):` / `chore(vN):` |

单文件 > 300 行或第二套职责 → 拆文件。单函数 > 80 行 → 拆步骤函数。
