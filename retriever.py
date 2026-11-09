"""Day 4：保留 Day 3 的文档切块，新增可切换的语义检索。"""
from pathlib import Path
import re
import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "knowledge"
SUPPORTED_SUFFIXES = {".md", ".txt"}
MAX_CHARS = 240
MODEL_NAME = "BAAI/bge-small-zh-v1.5"
_model = None

def pairs(text: str) -> set[str]:
    """提取中文/英文连续双字，用于原有关键词检索。"""
    parts = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", text.lower())
    return {part[i:i + 2] for part in parts for i in range(max(1, len(part) - 1))}

def load_documents() -> list[dict]:
    """读取 knowledge 第一层 Markdown/TXT，保留文件元数据。"""
    documents = []
    if not DATA_DIR.exists():
        return documents
    for path in sorted(DATA_DIR.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeError, OSError) as exc:
            print(f"[文档跳过] {path.name} | {type(exc).__name__}")
            continue
        if not content.strip():
            continue
        documents.append({
            "source": path.name, "text": content,
            "file_type": path.suffix.lower(),
            "line_count": len(content.splitlines()),
            "char_count": len(content),
        })
    return documents

def split_document(document: dict, max_chars: int = MAX_CHARS) -> list[dict]:
    """按空行分段；超长段按字符数切开。"""
    if max_chars < 1:
        raise ValueError("max_chars 必须大于 0")
    lines = document["text"].splitlines()
    blocks, buffer = [], []
    start_line = 1

    def flush(end_line: int) -> None:
        """写出当前段落，必要时继续拆小。"""
        if not buffer:
            return
        paragraph = "\n".join(buffer).strip()
        if paragraph:
            for offset in range(0, len(paragraph), max_chars):
                blocks.append({
                    "text": paragraph[offset:offset + max_chars],
                    "line_start": start_line, "line_end": end_line,
                })
        buffer.clear()

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            flush(line_number - 1)
            continue
        if not buffer:
            start_line = line_number
        buffer.append(line)
    flush(len(lines))
    return [
        {
            "chunk_id": f"{document['source']}#chunk_{i:03d}",
            "source": document["source"],
            "file_type": document["file_type"],
            "line_count": document["line_count"],
            "char_count": document["char_count"],
            **block,
        }
        for i, block in enumerate(blocks, start=1)
    ]

def load_chunks(max_chars: int = MAX_CHARS) -> list[dict]:
    """从所有文档生成可检索片段。"""
    chunks = []
    for doc in load_documents():
        chunks.extend(split_document(doc, max_chars=max_chars))
    return chunks

def keyword_retrieve(question: str, top_k: int = 2) -> list[dict]:
    """沿用 Day 3 的双字重合分数，作为对照基线。"""
    q = pairs(question.strip())
    if not q or top_k < 1:
        return []
    results = []
    for chunk in load_chunks():
        score = len(q & pairs(chunk["text"]))
        if score >= 2:
            results.append({**chunk, "score": score, "method": "keyword"})
    return sorted(
        results,
        key=lambda x: (-x["score"], x["source"], x["chunk_id"])
    )[:top_k]

def get_embedder():
    """首次使用语义检索时才加载模型，避免关键词模式额外下载。"""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model

def rank_embeddings(question_vector, chunk_vectors, chunks, top_k=2):
    """归一化向量的点积就是余弦相似度；可用假向量离线测试。"""
    if top_k < 1 or not chunks:
        return []
    q = np.asarray(question_vector, dtype=float).reshape(-1)
    matrix = np.asarray(chunk_vectors, dtype=float)
    if matrix.shape != (len(chunks), q.size):
        raise ValueError("向量维度或数量不匹配")
    q_norm = np.linalg.norm(q)
    c_norms = np.linalg.norm(matrix, axis=1)
    if q_norm == 0:
        return []
    scores = (matrix @ q) / (np.maximum(c_norms, 1e-12) * q_norm)
    order = sorted(
        range(len(chunks)),
        key=lambda i: (-float(scores[i]), chunks[i]["chunk_id"])
    )
    return [
        {**chunks[i], "score": round(float(scores[i]), 4), "method": "semantic"}
        for i in order[:top_k] if c_norms[i] > 0
    ]

def semantic_retrieve(question: str, top_k: int = 2) -> list[dict]:
    """将问题和知识片段编码为向量后比较语义相似度。"""
    if not question.strip() or top_k < 1:
        return []
    chunks = load_chunks()
    if not chunks:
        return []
    embedder = get_embedder()
    # BGE中文模型检索建议给问题加检索指令，文档保持原文。
    instruction = "为这个句子生成表示以用于检索相关文章："
    question_vector = embedder.encode([instruction + question])[0]
    chunk_vectors = embedder.encode([c["text"] for c in chunks])
    return rank_embeddings(question_vector, chunk_vectors, chunks, top_k)

def retrieve(question: str, top_k: int = 2, method: str = "keyword") -> list[dict]:
    """兼容旧调用：默认关键词；指定 semantic 才使用Embedding。"""
    if method == "keyword":
        return keyword_retrieve(question, top_k)
    if method == "semantic":
        return semantic_retrieve(question, top_k)
    raise ValueError("method 仅支持 keyword 或 semantic")