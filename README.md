# FinAgent Onion Learning

单仓叠加的金融分析 Agent 学习工程。当前版本：**V2 最小 RAG**。

## 环境

已有 conda 环境 `py12`，**不要**再 `python -m venv`。

```powershell
conda activate py12
cd ./finagent_onion_learning
pip install -r requirements.txt   # 首次或缺包时
python -V                         # 确认走的是 py12
```

确认根目录 `.env` 已配置 `DASHSCOPE_API_KEY`（可参考 `.env.example`）。

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

## 测试

```powershell
pytest tests\ingest\test_chunking.py tests\rag\test_retriever.py -q
python -c "import json,glob; [print(p,len(json.load(open(p,encoding='utf-8')))) for p in glob.glob('data/chunks/*.json')]"
```
