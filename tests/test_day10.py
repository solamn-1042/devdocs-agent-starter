"""Day10：不访问真实模型的离线测试。"""
import time
import unittest
from rerank_day10 import rerank, safe_search

def h(cid, text):
    return {"chunk_id": cid, "source": "demo.md", "text": text}

class TestDay10(unittest.TestCase):
    def test_rerank(self):
        self.assertEqual(rerank("API-404", [h("a", "天气"), h("b", "API-404")])[0]["chunk_id"], "b")

    def test_success(self):
        result = safe_search("API-404", search_fn=lambda q, top_k:
                             [h("a", "天气"), h("b", "API-404")])
        self.assertEqual(result["status"], "reranked")

    def test_error_fallback(self):
        def broken(q, hits, k):
            raise ValueError("模拟重排失败")
        result = safe_search("问题", search_fn=lambda q, top_k: [h("a", "原始证据")],
                             rerank_fn=broken)
        self.assertEqual(result["status"], "rerank_failed")
        self.assertEqual(result["hits"][0]["chunk_id"], "a")

    def test_timeout(self):
        def slow(q, hits, k):
            time.sleep(0.08)
            return hits
        result = safe_search("问题", timeout_s=0.005,
                             search_fn=lambda q, top_k: [h("a", "证据")],
                             rerank_fn=slow)
        self.assertEqual(result["status"], "timeout")

    def test_search_error(self):
        def broken(q, top_k):
            raise RuntimeError("检索故障")
        self.assertEqual(safe_search("问题", search_fn=broken)["status"], "search_failed")

    def test_empty(self):
        self.assertEqual(safe_search("问题", search_fn=lambda q, top_k: [])["status"], "empty")

if __name__ == "__main__":
    unittest.main()