"""Day9：复用Day8人工gold，对照Chroma与混合检索的Hit@3。"""
import csv
from pathlib import Path
from eval_day8 import load_cases, evaluate
from hybrid_search import hybrid_search
from vector_store import search_index, sync_index

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "evals" / "day9_comparison.csv"


def main():
    """同一套20题和同一Chroma索引，分别执行两种检索。"""
    cases = load_cases()  # 使用你已修复的Day8路径：evals/day8_cases.json
    print("同步索引：", sync_index())
    baseline = evaluate(cases, k=3, search_fn=search_index)
    mixed = evaluate(cases, k=3, search_fn=hybrid_search)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "question", "vector_hit", "hybrid_hit",
                         "vector_ms", "hybrid_ms", "vector_sources",
                         "hybrid_sources"])
        for old, new in zip(baseline, mixed):
            writer.writerow([old["id"], old["question"], old["hit"],
                             new["hit"], old["latency_ms"], new["latency_ms"],
                             old["top_sources"], new["top_sources"]])
    a = sum(r["hit"] for r in baseline)
    b = sum(r["hit"] for r in mixed)
    print(f"Chroma Hit@3: {a}/20 = {a / 20:.1%}")
    print(f"Hybrid Hit@3: {b}/20 = {b / 20:.1%}")
    print("对照CSV：", OUT)
    print("注意：如果混合检索分数下降，保留真实结果并检查Badcase。")


if __name__ == "__main__":
    main()