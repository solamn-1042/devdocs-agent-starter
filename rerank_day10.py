"""Day10：重排混合检索候选，并在失败时回退。"""
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from hybrid_search import hybrid_search, tokenize

def rerank(question, hits, top_k=3):
    """使用问题词项覆盖率排序，保留来源和Chunk ID。"""
    terms = set(tokenize(question))
    rows = []
    for rank, hit in enumerate(hits, 1):
        overlap = len(terms & set(tokenize(hit.get("text", "")))) / max(len(terms), 1)
        rows.append({**hit, "rerank_score": round(overlap + 0.01/rank, 6),
                     "original_rank": rank})
    return sorted(rows, key=lambda h: (-h["rerank_score"], h["original_rank"]))[:top_k]

def safe_search(question, top_k=3, timeout_s=2.0,
                search_fn=hybrid_search, rerank_fn=rerank):
    """检索失败返回空；重排失败或超时保留原始候选。"""
    start = time.perf_counter()
    try:
        hits = search_fn(question, top_k=10)
    except Exception as exc:
        return {"hits": [], "status": "search_failed", "error": str(exc),
                "ms": round((time.perf_counter()-start)*1000, 1)}
    if not hits:
        return {"hits": [], "status": "empty", "error": "", "ms": 0}
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(rerank_fn, question, hits, top_k)
    try:
        result = future.result(timeout=timeout_s)
        status, error = "reranked", ""
    except Exception as exc:
        result = hits[:top_k]
        status = "timeout" if isinstance(exc, TimeoutError) else "rerank_failed"
        error = str(exc)
        future.cancel()
    finally:
        # 超时后不阻塞调用方；线程可能仍短暂运行。
        pool.shutdown(wait=False, cancel_futures=True)
    return {"hits": result, "status": status, "error": error,
            "ms": round((time.perf_counter()-start)*1000, 1)}

def search_for_eval(question, top_k=3):
    """兼容Day8 evaluate()的search_fn参数。"""
    return safe_search(question, top_k=top_k)["hits"]

if __name__ == "__main__":
    q = input("请输入自然语言问题：").strip()
    result = safe_search(q)
    print("状态：", result["status"], "耗时ms：", result["ms"])
    for hit in result["hits"]:
        print(hit.get("source"), hit.get("chunk_id"),
              hit.get("rerank_score", "降级原始排名"))
        print(str(hit.get("text", ""))[:180])