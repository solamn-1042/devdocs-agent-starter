"""Day6：只测试证据门禁和引用校验，不调用模型、不访问Chroma。"""
import unittest
from unittest.mock import patch

from evidence import (
    assess_evidence,
    build_evidence_context,
    validate_citations,
)
from llm import answer


def make_hit(method="chroma", score=0.72):
    """构造一条与真实检索结果字段一致的测试证据。"""
    return {
        "source": "agent_practice.md",
        "chunk_id": "agent_practice.md#chunk_003",
        "line_start": 12,
        "line_end": 18,
        "file_type": ".md",
        "text": "工具超时后应检查请求状态，并在允许时进行有限次数重试。",
        "method": method,
        "score": score,
    }


class TestDay6Evidence(unittest.TestCase):
    def test_empty_hits_are_rejected(self):
        self.assertFalse(assess_evidence([])["ok"])

    def test_vector_score_below_threshold_is_rejected(self):
        result = assess_evidence([make_hit(score=0.20)])
        self.assertFalse(result["ok"])
        self.assertIn("低于当前门槛", result["reason"])

    def test_vector_score_at_threshold_passes(self):
        self.assertTrue(assess_evidence([make_hit(score=0.35)])["ok"])

    def test_keyword_hit_uses_keyword_rule_not_vector_threshold(self):
        hit = make_hit(method="keyword", score=3)
        self.assertTrue(assess_evidence([hit])["ok"])

    def test_missing_required_fields_are_ignored(self):
        self.assertFalse(assess_evidence([{"source": "x.md", "text": "内容"}])["ok"])

    def test_context_contains_source_chunk_and_lines(self):
        context = build_evidence_context([make_hit()])
        self.assertIn("agent_practice.md#chunk_003", context)
        self.assertIn("agent_practice.md", context)
        self.assertIn("12-18", context)

    def test_valid_citation_passes(self):
        hit = make_hit()
        result = validate_citations(
            "工具超时应有限重试。[来源: agent_practice.md#chunk_003]",
            [hit],
        )
        self.assertTrue(result["ok"])

    def test_missing_citation_fails(self):
        result = validate_citations("工具超时应有限重试。", [make_hit()])
        self.assertFalse(result["ok"])

    def test_unknown_chunk_citation_fails(self):
        result = validate_citations(
            "答案。[来源: unknown.md#chunk_999]",
            [make_hit()],
        )
        self.assertFalse(result["ok"])
        self.assertEqual(result["invalid_ids"], ["unknown.md#chunk_999"])

    def test_multiple_valid_citations_pass(self):
        first = make_hit()
        second = {
            **make_hit(),
            "chunk_id": "rag.md#chunk_001",
            "source": "rag.md",
        }
        result = validate_citations(
            "结论一。[来源: agent_practice.md#chunk_003] "
            "结论二。[来源: rag.md#chunk_001]",
            [first, second],
        )
        self.assertTrue(result["ok"])

    def test_no_evidence_does_not_call_model(self):
        with patch("llm.OpenAI") as fake_client:
            result = answer("这个项目的数据库密码是什么？", [])
            fake_client.assert_not_called()
        self.assertIn("没有找到足够可靠的证据", result)


if __name__ == "__main__":
    unittest.main()