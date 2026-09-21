---
name: stock-analyst
description: 上市公司年报与指标分析；当用户问财务/股价时启用
version: 0.1.0
---
# 何时使用
- 营收、毛利率、同比、年报原文
- 用户问某公司股价区间涨跌、行情
- 提到：贵州茅台、五粮液、宁德时代、中国平安、海康威视

# 工具
- rag_search, financial_indicator, stock_price, calculator
- 调用财务/行情前必须先 `company_lookup`

# 推荐顺序
1. `company_lookup`：公司中文名 → 股票代码
2. 按问题类型选一：
   - 结构化指标 / 跨年对比 → `financial_indicator(symbol)`
   - 年报原文、战略、风险等文本 → `rag_search(query)`（query 用短财务术语；公司与年份走 stock_code/year）
   - 某段时间股价/涨跌幅 → `stock_price(symbol, start_date, end_date)`，日期 YYYYMMDD
3. 需要算增长率、差值、百分比 → `calculator(expr)`，禁止心算

# 详细参考
见 references/tool-notes.md（仅在技能被选中后加载）
