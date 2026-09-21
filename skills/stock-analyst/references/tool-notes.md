# stock-analyst 工具备注（L2）

仅在调用 `read_skill_resource("stock-analyst", "references/tool-notes.md")` 时进入上下文。

## 年报 RAG 覆盖范围

| 公司 | 代码 | 年报年份（索引内） |
|------|------|-------------------|
| 贵州茅台 / 茅台 | 600519 | 2021–2025 |
| 五粮液 | 000858 | 2021–2025 |
| 宁德时代 | 300750 | 2021–2025 |
| 中国平安 / 平安 | 601318 | 2021–2025 |
| 海康威视 / 海康 | 002415 | 2021–2025 |

索引路径：项目根下 `vectorstore/faiss_index.bin` + `faiss_meta.json`。

## 工具选型

| 用户意图 | 优先工具 | 说明 |
|----------|----------|------|
| 「茅台股票代码」 | `company_lookup` | 静态映射，不联网 |
| 「近三年营收/毛利率/ROE」 | `financial_indicator` | AkShare 摘要表 |
| 「年报怎么写战略/风险」 | `rag_search` | 语义检索段落 |
| 「2023 年股价涨了多少」 | `stock_price` | 需起止日期 YYYYMMDD |
| 「同比增长率」 | 先取数再 `calculator` | 如 `(747-524)/524*100` |

## 调用示例

```text
1) company_lookup(name="茅台")           → 600519
2) financial_indicator(symbol="600519") → 近三年关键指标
   或 rag_search(query="毛利率", stock_code="600519", year="2023")
3) calculator(expr="(a-b)/b*100")         → 增长率%
```

股价示例：

```text
stock_price(symbol="600519", start_date="20230101", end_date="20231231")
```

## 降级说明

- 缺 AkShare / 网络失败：据错误信息如实告知，不要编造数字
- 缺 FAISS 索引或 Embedding Key：`rag_search` 会报错；可改用 `financial_indicator`
