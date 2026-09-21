# FinAgent Onion Learning

单仓叠加的金融分析 Agent 学习工程。当前版本：**V7 Skill 热拔插 + 渐进披露 + Harness**。

## 环境

已有 conda 环境 `py12`，**不要**再 `python -m venv`。

```powershell
conda activate py12
cd ./finagent_onion_learning
pip install -r requirements.txt   # 首次或缺包时（V6 新增 apscheduler）
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

## V5 运行

多步 ReAct：`company_lookup` → 年报 RAG / AkShare 指标与行情 → AST 计算器。可对照手写 Prompt 解析（`manual`）与原生 Function Calling（`fc`）。

先确保已建 V2 索引，并 `pip install akshare`。

```powershell
python src\agent\loop.py --mode fc --q "对比茅台与五粮液近三年毛利率，并计算差值"
python src\agent\loop.py --mode manual --q "分析一下贵州茅台2024年年报中提到的一些重要承诺事项？"
uvicorn src.agent.serve:app --reload --port 8012
python src\agent\evaluate.py
```

浏览器打开 `http://127.0.0.1:8012` 可看逐步轨迹。产物：`output/evaluate_v5.json`。

## V6 运行

跨会话记忆：七类 Markdown（`memory/`）+ SQLite 会话 + FAISS/FTS5 混合检索 + Memory Flush。对话中告诉 Agent 偏好，`/flush` 或满 20 条后新开会话应仍记得。

```powershell
uvicorn src.agent.serve:app --port 8013
python -c "from src.memory_sys.loader import load_base_prompt; assert len(load_base_prompt())>0"
```

浏览器打开 `http://127.0.0.1:8013`：默认「记忆对话」；也可切回 V5 ReAct 轨迹（会把 USER/MEMORY 拼进 system prompt）。HEARTBEAT 默认开启调度器，示例任务 `enabled: false`。

## V7 运行

Skill 目录热加载：启动只把 `name/description` 当 L0 索引；`load_skill` 后再展开 SKILL 正文并解锁 `tools.py`。Harness = 停止条件 + 工具白名单 + 可观测（轨迹里带 `activated_skills`）。

```powershell
python -c "from src.harness.skill_loader import load_skill_index; print(load_skill_index())"
python src\agent\loop.py --q "用股票分析技能解读宁德时代风险因素"
python src\harness\mcp_server.py --smoke
uvicorn src.agent.serve:app --port 8014
```

浏览器打开后可用 `GET /skills`、`POST /reload` 热拔插。新增 `skills/<name>/SKILL.md`（可选 `tools.py`）不改 agent 核心即可被发现。

## 测试

```powershell
pytest tests\ingest\test_chunking.py tests\rag\test_retriever.py tests\rag\test_rrf.py tests\tools\test_calculator_sandbox.py tests\memory_sys\test_loader.py tests\harness\test_hot_reload.py -q
python -c "from src.tools.rag_backend import rag_search; assert rag_search('测试')['ok'] in (True, False)"
python -c "from src.memory_sys.loader import load_base_prompt; assert len(load_base_prompt())>0"
python -c "from src.harness.skill_loader import load_skill_index; assert any(s['name']=='stock-analyst' for s in load_skill_index())"
python -c "import json,glob; [print(p,len(json.load(open(p,encoding='utf-8')))) for p in glob.glob('data/chunks/*.json')]"
```
