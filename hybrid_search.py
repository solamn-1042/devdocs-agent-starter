"""Day9：纯Python BM25 + 已有Chroma结果，使用RRF融合。无新依赖。"""
import math
import re
from collections import Counter
from retriever import load_chunks
from vector_store import search_index


def tokenize(text):
    """英文/编号保留完整词；中文取连续双字，兼顾短语与专业编号。"""
    parts = re.findall(r"[a-zA-Z0-9_.-]+|[\u4e00-\u9fff]+", text.lower())
    tokens = []
    for part in parts:
        if re.fullmatch(r"[\u4e00-\u9fff]+", part):
            tokens.extend(part[i:i + 2] for i in range(len(part) - 1))
            if len(part) == 1:
                tokens.append(part)
        else:
            tokens.append(part)
    return tokens


def bm25_search(question, top_k=10, chunks=None):
    """对现有知识库Chunk计算BM25；仅返回有词项重合的候选。"""
    if not question.strip() or top_k < 1:
        return []
    chunks = load_chunks() if chunks is None else chunks
    if not chunks:
        return []
    docs = [tokenize(chunk["text"]) for chunk in chunks]
    query_terms = set(tokenize(question))
    if not query_terms:
        return []
    lengths = [len(doc) for doc in docs]
    avg_len = max(sum(lengths) / len(docs), 1)
    df = Counter(term for doc in docs for term in set(doc))
    n = len(docs)
    results = []
    for chunk, doc, length in zip(chunks, docs, lengths):
        freq = Counter(doc)
        score = 0.0
        for term in query_terms:
            if not freq[term]:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            score += idf * freq[term] * 2.2 / (
                freq[term] + 1.2 * (0.25 + 0.75 * length / avg_len)
            )
        if score > 0:
            results.append({**chunk, "score": round(score, 4), "method": "bm25"})
    return sorted(results, key=lambda h: (-h["score"], h["chunk_id"]))[:top_k]


def rrf_fuse(vector_hits, bm25_hits, top_k=3, rrf_k=60):
    """按Chunk ID合并两路排名；返回完整证据字段供原有页面/评测使用。"""
    combined = {}
    for name, hits in (("vector", vector_hits), ("bm25", bm25_hits)):
        for rank, hit in enumerate(hits, start=1):
            key = hit["chunk_id"]
            if key not in combined:
                combined[key] = {**hit, "score": 0.0, "method": "hybrid",
                                 "vector_rank": None, "bm25_rank": None}
            combined[key]["score"] += 1.0 / (rrf_k + rank)
            combined[key][name + "_rank"] = rank
    return sorted(combined.values(),
                  key=lambda h: (-h["score"], h["chunk_id"]))[:top_k]


def hybrid_search(question, top_k=3, candidate_k=10, vector_fn=search_index,
                  chunks=None):
    """真实向量Top10与本地BM25 Top10融合；向量失败时返回BM25候选。"""
    if not question.strip() or top_k < 1:
        return []
    candidate_k = max(candidate_k, top_k)
    lexical = bm25_search(question, top_k=candidate_k, chunks=chunks)
    try:
        vectors = vector_fn(question, top_k=candidate_k)
    except Exception as exc:
        print(f"[向量检索失败，使用BM25] {type(exc).__name__}: {exc}")
        vectors = []
    return rrf_fuse(vectors, lexical, top_k=top_k)