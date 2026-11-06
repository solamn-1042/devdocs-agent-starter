"""第一课：非常简单的本地关键词检索（后续逐步替换成Embedding）。"""
from pathlib import Path
import re

# 使用相对当前文件的路径，不依赖PowerShell处在哪个目录。
DATA_DIR = Path(__file__).resolve().parent / "knowledge"

def pairs(text: str) -> set[str]:
    """提取连续2字符片段，兼顾中文词和英文字母；这不是语义向量。"""
    parts = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", text.lower())
    return {part[i:i+2] for part in parts for i in range(max(1, len(part)-1))}

def retrieve(question: str, top_k: int = 2) -> list[dict]:
    """返回相似的Markdown文档及其分数；不匹配时返回空列表。"""
    q = pairs(question.strip())
    if not q or top_k < 1:
        return []
    results = []
    for path in sorted(DATA_DIR.glob("*.md")):
        content = path.read_text(encoding="utf-8")
        # 分数表示两个字符连续片段的交集数量，不是准确率。
        score = len(q & pairs(content))
        if score >= 2:
            results.append({"source": path.name, "text": content, "score": score})
    return sorted(results, key=lambda item: (-item["score"], item["source"]))[:top_k]
