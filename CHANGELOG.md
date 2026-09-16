# CHANGELOG

每版验收、打 tag 并推送 GitHub 后，在这里追加一节。对照命令：

```text
git diff v{N-1}.0..vN.0 --stat
# GitHub: https://github.com/USER/REPO/compare/v{N-1}.0...vN.0
```

## v1.0 — 解析与分块

- 相对 tag：`v0.0`
- 意图：Loader → 结构化块 → fixed / semantic / hierarchical，落盘 JSON
- 关键文件：`src/ingest/parse_*.py` `src/ingest/chunking.py` `src/ingest/run_chunk.py` `tests/ingest/test_chunking.py`

## v0.0 — 脚手架

- 相对 tag：仓库初始化
- 意图：单仓目录、环境、契约
- 关键文件：`src/hello_agent.py` `src/common/config.py` `src/common/types.py` `shared/CONVENTIONS.md`
