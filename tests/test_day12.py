"""Day12：离线测试工具执行边界，不发模型API请求。"""
import unittest
from unittest.mock import patch
from agent_day12 import dispatch, build_graph, TOOLS

def fake_search(query, top_k=3):
    """模拟Day10 safe_search的真实返回结构。"""
    return {"status": "reranked", "hits": [
        {"source": "knowledge/demo.md", "chunk_id": "c001",
         "text": "索引更新后应重新同步Chroma。"}]}

class Day12Tests(unittest.TestCase):
    def test_tools_registered(self):
        self.assertEqual({t["function"]["name"] for t in TOOLS},
                         {"search_docs", "read_source"})

    def test_invalid_arguments(self):
        self.assertEqual(dispatch("delete_all", "{}", {})["error"],
                         "tool_not_allowed")
        self.assertEqual(dispatch("search_docs", "{bad", {})["error"],
                         "invalid_json")
        self.assertEqual(dispatch("search_docs", '{"query":"x","path":"."}', {})["error"],
                         "invalid_fields")
        self.assertEqual(dispatch("search_docs", '{"query":""}', {})["error"],
                         "invalid_value")

    def test_search_then_read(self):
        cache = {}
        with patch("agent_day12.safe_search", side_effect=fake_search):
            found = dispatch("search_docs", '{"query":"索引如何更新"}', cache)
        self.assertEqual(found["hits"][0]["chunk_id"], "c001")
        detail = dispatch("read_source", '{"chunk_id":"c001"}', cache)
        self.assertIn("Chroma", detail["text"])

    def test_read_requires_prior_search(self):
        self.assertEqual(dispatch("read_source", '{"chunk_id":"c001"}', {})["error"],
                         "chunk_not_in_current_search")

    def test_graph_compiles(self):
        self.assertIsNotNone(build_graph())

if __name__ == "__main__":
    unittest.main()