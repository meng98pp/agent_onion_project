## 类型契约

- Chunk 与 ToolResult 的 Python 定义见 `src/common/types.py`
- 领域模块只依赖这些类型，不依赖对方的私有类

## Block（parse_* 产出，chunk_* 只消费这个结构）

```json
{
  "block_type": "text|table|title",
  "content": "string",
  "page_num": 1,
  "section_path": ["第三章 管理层讨论", "一、经营情况概述"],
  "source": "demo.txt",
  "is_ocr": false
}
```

## Chunk

```json
{
  "chunk_id": "string",
  "text": "string",
  "source": "string",
  "page_num": 1,
  "section_path": "string",
  "strategy": "fixed|semantic|hierarchical",
  "parent_id": null,
  "block_types": ["text"],
  "stock_code": "600000",
  "year": "2023"
}
```

## Tool Result

```json
{ "ok": true, "data": {}, "error": null }
```

## 版本策略（单仓叠加）

- 只有一套 `src/`：按领域目录叠加，不平行复制工程
- 规范见 `shared/CONVENTIONS.md`
- 每版验收后打 tag `vN.0`，用 `git diff v{N-1}.0..vN.0` 看本层改动
- API Key 只放 `.env`，只通过 `src/common/config.py` 读取

