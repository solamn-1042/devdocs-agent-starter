"""Day8：固定20条人工标注检索评测；仅调用本地Chroma检索，不调用LLM。"""
import csv
import json
import time
from pathlib import Path

from vector_store import search_index, sync_index

ROOT = Path(__file__).resolve().parent
CASES_PATH = ROOT / "evals" / "day8_cases.json"
REPORT_PATH = ROOT / "evals" / "day8_results.csv"
BADCASE_PATH = ROOT / "evals" / "day8_badcases.md"


def load_cases():
    """加载人工标注的问题、正确来源和应出现的原文句子。"""
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    if len(cases) != 20 or len({c["id"] for c in cases}) != 20:
        raise ValueError("评测集必须有20条不同ID的问题。")
    for case in cases:
        if not all(case.get(k) for k in ("id", "question", "gold_source", "gold_text")):
            raise ValueError(f"标注字段缺失：{case}")
    return cases


def matches_gold(hit, case):
    """同时检查来源和证据正文；避免只命中文件名就误判正确。"""
    return (
        Path(str(hit.get("source", ""))).name == case["gold_source"]
        and case["gold_text"] in str(hit.get("text", ""))
    )


def evaluate(cases, k=3, search_fn=search_index):
    """对每个问题取前K个候选，计算Hit@K并保留失败详情。"""
    rows = []
    for case in cases:
        started = time.perf_counter()
        try:
            hits = search_fn(case["question"], top_k=k)
            error = ""
        except Exception as exc:
            hits = []
            error = f"{type(exc).__name__}: {exc}"
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        found = any(matches_gold(hit, case) for hit in hits)
        rows.append({
            "id": case["id"],
            "question": case["question"],
            "hit": int(found),
            "latency_ms": elapsed_ms,
            "gold_source": case["gold_source"],
            "gold_text": case["gold_text"],
            "top_sources": " | ".join(str(h.get("source", "")) for h in hits),
            "top_chunks": " | ".join(str(h.get("chunk_id", "")) for h in hits),
            "error": error,
        })
    return rows


def save_reports(rows):
    """保存可复核的CSV和只包含失败案例的Markdown。"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    failed = [row for row in rows if not row["hit"]]
    lines = ["# Day8 检索Badcase", "", "只记录本机真实运行结果。", ""]
    for row in failed:
        lines.extend([
            f"## {row['id']}：{row['question']}",
            f"- 预期来源：{row['gold_source']}",
            f"- 预期证据：{row['gold_text']}",
            f"- 实际来源：{row['top_sources'] or '无'}",
            f"- 实际Chunk：{row['top_chunks'] or '无'}",
            f"- 异常：{row['error'] or '无'}",
            "- 待分析原因：",
            "",
        ])
    BADCASE_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    """同步知识库，执行20条评测并打印真实统计。"""
    cases = load_cases()
    print("先同步本地Chroma索引……")
    print("同步报告：", sync_index())
    rows = evaluate(cases, k=3)
    save_reports(rows)
    hit_count = sum(row["hit"] for row in rows)
    avg_ms = sum(row["latency_ms"] for row in rows) / len(rows)
    print(f"Hit@3 = {hit_count}/20 = {hit_count / 20:.1%}")
    print(f"平均检索耗时 = {avg_ms:.1f} ms（仅本机本次测量）")
    for row in rows:
        print(f"{row['id']} {'命中' if row['hit'] else '未命中'} | {row['question']}")
    print("完整记录：", REPORT_PATH)
    print("失败案例：", BADCASE_PATH)


if __name__ == "__main__":
    main()