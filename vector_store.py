"""Day5：Chroma 本地持久化，索引更新与语义检索。"""
from pathlib import Path
import hashlib
import json
from retriever import load_chunks, get_embedder

DB_DIR = Path(__file__).resolve().parent / "chroma_db"
COLLECTION_NAME = "devdocs_v1"


def get_collection():
    """按需导入 Chroma，首次创建后在本地磁盘保存。"""
    import chromadb

    client = chromadb.PersistentClient(path=str(DB_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )


def chunk_key(chunk):
    """以来源、分块编号和正文摘要生成确定性ID；正文变化后ID变化。"""
    raw = f"{chunk['source']}\0{chunk['chunk_id']}\0{chunk['text']}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def fingerprint(chunks):
    """对当前文档快照生成摘要，便于定位更新是否发生。"""
    raw = json.dumps(
        [(c["source"], c["chunk_id"], c["text"]) for c in chunks], ensure_ascii=False
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def sync_index(collection=None, chunks=None, embedder=None):
    """仅编码新增或变更块，删除已不存在的块，重复执行不重复写入。"""
    collection = collection if collection is not None else get_collection()
    chunks = chunks if chunks is not None else load_chunks()
    expected = {chunk_key(c): c for c in chunks}
    existing = set(collection.get(include=[])["ids"])
    removed = sorted(existing - set(expected))
    added = sorted(set(expected) - existing)
    if removed:
        collection.delete(ids=removed)
    if added:
        model = embedder if embedder is not None else get_embedder()
        texts = [expected[k]["text"] for k in added]
        vectors = model.encode(texts)
        collection.upsert(
            ids=added,
            documents=texts,
            embeddings=[
                v.tolist() if hasattr(v, "tolist") else list(v) for v in vectors
            ],
            metadatas=[
                {
                    "source": expected[k]["source"],
                    "chunk_id": expected[k]["chunk_id"],
                    "line_start": int(expected[k]["line_start"]),
                    "line_end": int(expected[k]["line_end"]),
                    "file_type": expected[k]["file_type"],
                }
                for k in added
            ],
        )
    return {
        "total": len(expected),
        "added": len(added),
        "removed": len(removed),
        "unchanged": len(expected) - len(added),
        "fingerprint": fingerprint(chunks),
    }


def search_index(question, top_k=3, collection=None, embedder=None):
    """实时只编码问题，查询磁盘索引并还原原有检索结果字段。"""
    if not question.strip() or top_k < 1:
        return []
    collection = collection if collection is not None else get_collection()
    if collection.count() == 0:
        return []
    model = embedder if embedder is not None else get_embedder()
    instruction = "为这个句子生成表示以用于检索相关文章："
    vec = model.encode([instruction + question])[0]
    vec = vec.tolist() if hasattr(vec, "tolist") else list(vec)
    data = collection.query(
        query_embeddings=[vec],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for text, meta, distance in zip(
        data["documents"][0], data["metadatas"][0], data["distances"][0]
    ):
        hits.append(
            {
                **meta,
                "text": text,
                "score": round(1.0 - float(distance), 4),
                "method": "chroma",
            }
        )
    return hits
