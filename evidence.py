"""Day6：证据门禁、上下文整理与引用校验。纯Python实现，不调用模型。"""
import re

# 这是一个起始阈值，不是通用正确率标准；后续要用固定测试集校准。
MIN_VECTOR_SCORE = 0.35

# 模型必须使用这个格式引用检索结果中的 Chunk ID。
CITATION_PATTERN = re.compile(r"\[来源:\s*([^\]\s]+)\s*\]")

# 证据不足时统一使用明确拒答，避免继续调用模型补写事实。
REFUSAL_MESSAGE = "知识库中没有找到足够可靠的证据，暂时无法回答。请补充相关文档或换一种问法。"


def usable_hits(hits):
    """只保留同时具有来源、Chunk ID 和正文的检索结果。"""
    result = []
    for hit in hits or []:
        if not isinstance(hit, dict):
            continue
        if not hit.get("source") or not hit.get("chunk_id") or not hit.get("text"):
            continue
        result.append(hit)
    return result


def assess_evidence(hits, min_vector_score=MIN_VECTOR_SCORE):
    """判断是否有可用证据；向量检索额外检查最高余弦相似度。"""
    usable = usable_hits(hits)
    if not usable:
        return {
            "ok": False,
            "reason": "没有可用的文档片段。",
            "hits": [],
            "best_score": None,
        }

    # Day5 的 Chroma 分数和 Day4 的语义分数都是余弦相似度；
    # 关键词分数是字符重合数量，量纲不同，不能套用同一个阈值。
    vector_hits = [
        hit for hit in usable
        if hit.get("method") in {"chroma", "semantic"}
    ]

    if vector_hits:
        scores = []
        for hit in vector_hits:
            try:
                scores.append(float(hit.get("score")))
            except (TypeError, ValueError):
                continue

        best_score = max(scores, default=None)
        if best_score is None:
            return {
                "ok": False,
                "reason": "向量检索结果缺少有效相似度分数。",
                "hits": usable,
                "best_score": None,
            }

        if best_score < min_vector_score:
            return {
                "ok": False,
                "reason": (
                    f"最高向量相似度 {best_score:.4f} "
                    f"低于当前门槛 {min_vector_score:.2f}。"
                ),
                "hits": usable,
                "best_score": best_score,
            }

        return {
            "ok": True,
            "reason": f"向量证据门槛通过，最高相似度 {best_score:.4f}。",
            "hits": usable,
            "best_score": best_score,
        }

    # 关键词检索沿用 retriever.py 自己的最低重合条件；
    # 不把关键词计数误当成余弦相似度。
    return {
        "ok": True,
        "reason": "存在通过关键词检索条件的文档片段。",
        "hits": usable,
        "best_score": None,
    }


def build_evidence_context(hits):
    """把每个证据的来源、Chunk ID、行号和正文交给模型。"""
    blocks = []
    for hit in usable_hits(hits):
        source = hit["source"]
        chunk_id = hit["chunk_id"]
        line_start = hit.get("line_start", "?")
        line_end = hit.get("line_end", "?")
        blocks.append(
            f"[证据ID: {chunk_id}]\n"
            f"[文件: {source}] [行: {line_start}-{line_end}]\n"
            f"{hit['text']}"
        )
    return "\n\n---\n\n".join(blocks)


def validate_citations(answer_text, hits):
    """检查模型引用的 Chunk ID 是否真实存在于本次检索结果中。"""
    cited_ids = CITATION_PATTERN.findall(answer_text or "")
    allowed_ids = {
        str(hit["chunk_id"])
        for hit in usable_hits(hits)
    }

    if not cited_ids:
        return {
            "ok": False,
            "reason": "回答没有包含任何符合格式的来源引用。",
            "cited_ids": [],
            "invalid_ids": [],
        }

    invalid_ids = sorted(set(cited_ids) - allowed_ids)
    if invalid_ids:
        return {
            "ok": False,
            "reason": "回答引用了本次检索结果中不存在的 Chunk ID。",
            "cited_ids": cited_ids,
            "invalid_ids": invalid_ids,
        }

    return {
        "ok": True,
        "reason": "所有引用的 Chunk ID 均属于本次检索结果。",
        "cited_ids": cited_ids,
        "invalid_ids": [],
    }