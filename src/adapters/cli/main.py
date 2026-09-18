"""
main.py — fincli 风格子命令入口（argparse）

直接调用 src.tools，不含 LLM。stdout 输出可读文本供 Agent 侧 subprocess 回传。

用法：
  python src\\adapters\\cli\\main.py search --query "营收" --stock-code 600519 --year 2023
  python src\\adapters\\cli\\main.py list-companies
  python src\\adapters\\cli\\main.py weather --city 北京
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.tools.rag_backend import (  # noqa: E402
    list_companies,
    rag_search,
    tool_result_text as rag_text,
)
from src.tools.weather_backend import (  # noqa: E402
    query_weather,
    tool_result_text as weather_text,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="fincli",
        description="fincli — A股年报检索 + 天气查询",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_search = sub.add_parser("search", help="检索年报段落")
    p_search.add_argument("--query", required=True)
    p_search.add_argument("--stock-code", default=None)
    p_search.add_argument("--year", default=None)
    p_search.add_argument("--top-k", type=int, default=5)

    sub.add_parser("list-companies", help="列出知识库公司")

    p_weather = sub.add_parser("weather", help="查询城市天气")
    p_weather.add_argument("--city", required=True)

    args = parser.parse_args()

    if args.cmd == "search":
        print(rag_text(rag_search(args.query, args.top_k, args.stock_code, args.year)))
    elif args.cmd == "list-companies":
        print(rag_text(list_companies()))
    elif args.cmd == "weather":
        print(weather_text(query_weather(args.city)))


if __name__ == "__main__":
    main()
