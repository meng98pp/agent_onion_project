# FinAgent Onion Learning

单仓叠加的金融分析 Agent 学习工程。当前版本：**V1 解析 + 三种分块**。

## 环境

已有 conda 环境 `py12`，**不要**再 `python -m venv`。

```powershell
conda activate py12
cd ./finagent_onion_learning
pip install -r requirements.txt   # 首次或缺包时
python -V                         # 确认走的是 py12
```

## V1 运行

读取 `data/raw/` 下已有的 `.txt` / `.pdf`（含你拷入的财报），写出解析块和三种分块。

```powershell
python src\ingest\run_chunk.py
```

产物：

- `data/parsed/<stem>.json`：title / table / text 块
- `data/chunks/<stem>_{fixed,semantic,hierarchical}.json`：三种分块

## 测试

```powershell
pytest tests\ingest\test_chunking.py -q
python -c "import json,glob; [print(p,len(json.load(open(p,encoding='utf-8')))) for p in glob.glob('data/chunks/*.json')]"
```
