"""离线测试检索，不消耗模型API费用。"""
import unittest
from retriever import pairs, retrieve

class TestRetriever(unittest.TestCase):
    def test_pairs(self):
        self.assertIn("检索", pairs("文档检索来源"))

    def test_agent_question(self):
        hits = retrieve("Agent的工具调用由谁执行？")
        self.assertTrue(any(x["source"] == "01_agent.md" for x in hits))

    def test_rag_question(self):
        hits = retrieve("RAG如何引用来源？")
        self.assertTrue(any(x["source"] == "02_rag.md" for x in hits))

    def test_irrelevant_question(self):
        self.assertEqual(retrieve("火星天气预报如何查看"), [])

    def test_limits(self):
        self.assertLessEqual(len(retrieve("文档来源检索模型工具", top_k=2)), 2)

if __name__ == "__main__":
    unittest.main()
