"""Day10：用Day8固定20题对比混合检索与重排。"""
import csv
from pathlib import Path
from statistics import mean
from eval_day8 import load_cases, evaluate
from hybrid_search import hybrid_search
from rerank_day10 import search_for_eval
from vector_store import sync_index

def main():
    """保留真实Hit@3和耗时，输出逐题CSV。"""
    print("同步索引：", sync_index())
    cases = load_cases()
    before = evaluate(cases, k=3, search_fn=hybrid_search)
    after = evaluate(cases, k=3, search_fn=search_for_eval)
    path = Path(__file__).resolve().parent / "evals" / "day10_comparison.csv"
    path.parent.mkdir(exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "question", "hybrid_hit", "rerank_hit",
                         "hybrid_ms", "rerank_ms", "hybrid_sources", "rerank_sources"])
        for a, b in zip(before, after):
            writer.writerow([a["id"], a["question"], a["hit"], b["hit"],
                             a["latency_ms"], b["latency_ms"],
                             a["top_sources"], b["top_sources"]])
    print("Hybrid Hit@3:", sum(x["hit"] for x in before), "/20")
    print("Rerank Hit@3:", sum(x["hit"] for x in after), "/20")
    print("Hybrid平均ms:", round(mean(x["latency_ms"] for x in before), 1))
    print("Rerank平均ms:", round(mean(x["latency_ms"] for x in after), 1))
    print("CSV:", path)

if __name__ == "__main__":
    main()