"""Day 3：Markdown/TXT 加载、按段落分块、Chunk ID 与关键词检索。"""
from pathlib import Path
import re

DATA_DIR = Path(__file__).resolve().parent / "knowledge"
SUPPORTED_SUFFIXES = {".md", ".txt"}
MAX_CHARS = 240  # 单个片段最多字符数，先用固定值观察效果


def pairs(text: str) -> set[str]:
    """提取连续两个字符，沿用 Day 1 的简单关键词评分。"""
    parts = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", text.lower())
    return {
        part[i:i + 2]
        for part in parts
        for i in range(max(1, len(part) - 1))
    }


def load_documents() -> list[dict]:
    """读取 knowledge/ 第一层的 Markdown/TXT，保留 Day 2 的元数据。"""
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
            "source": path.name,
            "text": content,
            "file_type": path.suffix.lower(),
            "line_count": len(content.splitlines()),
            "char_count": len(content),
        })
    return documents


def split_document(document: dict, max_chars: int = MAX_CHARS) -> list[dict]:
    """按空行分段；超长段按字符切开，保留每块真实行号范围。"""
    if max_chars < 1:
        raise ValueError("max_chars 必须大于 0")
    lines = document["text"].splitlines()
    blocks = []
    buffer = []
    start_line = 1

    def flush(end_line: int) -> None:
        """把当前段落写入 blocks，超长时继续切成小块。"""
        if not buffer:
            return
        paragraph = "\n".join(buffer).strip()
        if paragraph:
            for offset in range(0, len(paragraph), max_chars):
                blocks.append({
                    "text": paragraph[offset:offset + max_chars],
                    "line_start": start_line,
                    "line_end": end_line,
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

    chunks = []
    for index, block in enumerate(blocks, start=1):
        chunks.append({
            "chunk_id": f"{document['source']}#chunk_{index:03d}",
            "source": document["source"],
            "file_type": document["file_type"],
            "line_count": document["line_count"],
            "char_count": document["char_count"],
            **block,
        })
    return chunks


def load_chunks(max_chars: int = MAX_CHARS) -> list[dict]:
    """将每份文档拆成可检索的独立片段。"""
    chunks = []
    for document in load_documents():
        chunks.extend(split_document(document, max_chars=max_chars))
    return chunks


def retrieve(question: str, top_k: int = 2) -> list[dict]:
    """给片段打分，返回 Top-K；无命中返回空列表。"""
    q = pairs(question.strip())
    if not q or top_k < 1:
        return []
    results = []
    for chunk in load_chunks():
        score = len(q & pairs(chunk["text"]))
        if score >= 2:
            results.append({**chunk, "score": score})
    return sorted(
        results,
        key=lambda item: (-item["score"], item["source"], item["chunk_id"])
    )[:top_k]