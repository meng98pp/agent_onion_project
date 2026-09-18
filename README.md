# FinAgent Onion Learning

单仓叠加的金融分析 Agent 学习工程。当前版本：**V4 工具调用（Function Call × MCP × CLI）**。

## 环境

已有 conda 环境 `py12`，**不要**再 `python -m venv`。

```powershell
conda activate py12
cd ./finagent_onion_learning
pip install -r requirements.txt   # 首次或缺包时（V4 新增 mcp）
python -V                         # 确认走的是 py12
```

确认根目录 `.env` 已配置 `DASHSCOPE_API_KEY`（可参考 `.env.example`）。可选 `DEEPSEEK_API_KEY`。

## V1 运行

读取 `data/raw/` 下已有的 `.txt` / `.pdf`（含你拷入的财报），写出解析块和三种分块。

```powershell
python src\ingest\run_chunk.py
```

产物：

- `data/parsed/<stem>.json`：title / table / text 块
- `data/chunks/<stem>_{fixed,semantic,hierarchical}.json`：三种分块

## V2 运行

默认索引 `*_semantic.json`，离线建 FAISS，再在线问答。

```powershell
python src\rag\indexer.py
python src\rag\ask.py "毛利率相关表述有哪些？"
python src\rag\ask.py "不存在的公司XYZ的营收"
```

产物：

- `vectorstore/faiss_index.bin`：FAISS `IndexFlatIP`
- `vectorstore/faiss_meta.json`：与向量行对齐的 chunk 元数据

## V3 运行

同一问题可切换 `vector_only` / `hybrid` / `hybrid_rerank`；评估默认 Hit@K，可选 RAGAS。

`hybrid_rerank` 使用本地 CrossEncoder：`models/bge-reranker-base/`（缺模型时自动跳过精排）。可用环境变量 `RERANK_MODEL` 覆盖目录名或绝对路径。

```powershell
python src\rag\ask.py --mode hybrid --q "贵州茅台2024年营业收入？"
python src\rag\ask.py --mode hybrid_rerank --q "贵州茅台2024年营业收入？"
python src\rag\evaluate_ragas.py
python src\rag\evaluate_ragas.py --ablation vector_only,hybrid,hybrid_rerank
python src\rag\evaluate_ragas.py --ragas   # 需已装 ragas/datasets，会额外调 LLM
```

产物：

- `evaluation/gold_qa.json`：金标题
- `evaluation/results/*.json`：Hit@K / 消融（及可选 RAGAS）结果

### v3.1说明
优化了解析、分块、评估，效果一般。进阶可使用MinerU/Marker/PaddleOCR/Qwen-VL等工具解析

## V4 运行

同一后端 `src/tools/`（`rag_search` / `query_weather`），三种适配器只接线：

```powershell
python src\adapters\function_call\run_fc.py "茅台2024营收"
python src\adapters\mcp\run_mcp_host.py --q "茅台2024营收"
python src\adapters\cli\run_cli_agent.py "北京天气"
python src\compare.py
```

纯 CLI 子命令（无 LLM）：

```powershell
python src\adapters\cli\main.py search --query "营收" --stock-code 600519 --year 2023
python src\adapters\cli\main.py weather --city 北京
```

产物：`output/compare_result.md`（同一问题 × 三方式对比表）。

## 测试

```powershell
pytest tests\ingest\test_chunking.py tests\rag\test_retriever.py tests\rag\test_rrf.py -q
python -c "from src.tools.rag_backend import rag_search; assert rag_search('测试')['ok'] in (True, False)"
python -c "import json,glob; [print(p,len(json.load(open(p,encoding='utf-8')))) for p in glob.glob('data/chunks/*.json')]"
```
