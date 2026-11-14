"""Day9：不访问Chroma和模型的离线融合测试。"""
import unittest
from hybrid_search import tokenize, bm25_search, rrf_fuse, hybrid_search


def hit(cid, text="证据"):
    return {"source": "demo.md", "chunk_id": cid, "text": text,
            "line_start": 1, "line_end": 1, "file_type": ".md"}


class TestDay9(unittest.TestCase):
    def test_tokens_keep_technical_id(self):
        self.assertIn("api-404", tokenize("接口 API-404 报错"))

    def test_bm25_prefers_exact_identifier(self):
        docs = [hit("a", "请求超时"), hit("b", "API-404 身份认证失败")]
        self.assertEqual(bm25_search("API-404", chunks=docs)[0]["chunk_id"], "b")

    def test_no_match_returns_empty(self):
        self.assertEqual(bm25_search("zzzxxyy", chunks=[hit("a")]), [])

    def test_rrf_shared_result_wins(self):
        a, b, c = hit("a"), hit("b"), hit("c")
        result = rrf_fuse([a, b], [c, a])
        self.assertEqual(result[0]["chunk_id"], "a")
        self.assertEqual(result[0]["method"], "hybrid")

    def test_hybrid_preserves_source(self):
        docs = [hit("a", "API-404 身份认证失败")]
        result = hybrid_search("API-404", chunks=docs,
                               vector_fn=lambda question, top_k: [])
        self.assertEqual(result[0]["source"], "demo.md")

    def test_vector_failure_falls_back(self):
        def broken(question, top_k):
            raise RuntimeError("模拟Chroma故障")
        result = hybrid_search("API-404",
                               chunks=[hit("a", "API-404 身份认证失败")],
                               vector_fn=broken)
        self.assertEqual(result[0]["chunk_id"], "a")


if __name__ == "__main__":
    unittest.main()