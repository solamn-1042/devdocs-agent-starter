"""Day 2：加载 Markdown/TXT，保留文件来源；继续使用 Day 1 的关键词算法。"""
from pathlib import Path
import re

DATA_DIR = Path(__file__).resolve().parent / "knowledge"
SUPPORTED_SUFFIXES = {".md", ".txt"}


def pairs(text: str) -> set[str]:
    """提取连续两个字符；本课不修改 Day 1 的匹配规则。"""
    parts = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", text.lower())
    return {
        part[i:i + 2]
        for part in parts
        for i in range(max(1, len(part) - 1))
    }


def load_documents() -> list[dict]:
    """扫描知识库，读取两种纯文本格式，并保留可追溯的元数据。"""
    documents = []
    for path in sorted(DATA_DIR.iterdir()):
        # 文件夹或不支持的后缀不参与检索。
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        try:
            # Markdown/TXT 都按 UTF-8 纯文本读取。
            content = path.read_text(encoding="utf-8")
        except (UnicodeError, OSError) as exc:
            # 某个文件损坏时跳过它，其他文件仍可使用。
            print(f"[文档跳过] {path.name} | {type(exc).__name__}")
            continue
        if not content.strip():
            # 空文件不应占据检索结果名额。
            continue
        documents.append({
            "source": path.name,
            "text": content,
            "file_type": path.suffix.lower(),
            "line_count": len(content.splitlines()),
            "char_count": len(content),
        })
    return documents


def retrieve(question: str, top_k: int = 2) -> list[dict]:
    """沿用原有关键词评分；命中结果额外携带文件类型和行数。"""
    q = pairs(question.strip())
    if not q or top_k < 1:
        return []
    results = []
    for document in load_documents():
        score = len(q & pairs(document["text"]))
        if score >= 2:
            results.append({**document, "score": score})
    return sorted(
        results,
        key=lambda item: (-item["score"], item["source"])
    )[:top_k]