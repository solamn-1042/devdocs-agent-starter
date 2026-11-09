"""Day 4：使用确定性假向量验证排序，不下载模型、不调用API。"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import retriever

class TestSemantic(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        patcher = patch.object(retriever, "DATA_DIR", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_cosine_order(self):
        chunks = [
            {"chunk_id": "a#001", "text": "A"},
            {"chunk_id": "b#001", "text": "B"},
        ]
        hits = retriever.rank_embeddings([1, 0], [[0, 1], [0.9, 0.1]], chunks)
        self.assertEqual(hits[0]["chunk_id"], "b#001")
        self.assertGreater(hits[0]["score"], hits[1]["score"])

    def test_top_k(self):
        chunks = [{"chunk_id": f"c#{i}", "text": str(i)} for i in range(3)]
        hits = retriever.rank_embeddings([1, 0], [[1, 0], [0, 1], [1, 1]], chunks, top_k=1)
        self.assertEqual(len(hits), 1)

    def test_empty(self):
        self.assertEqual(retriever.rank_embeddings([1, 0], [], []), [])
        self.assertEqual(retriever.retrieve("", method="semantic"), [])

    def test_invalid_dimension(self):
        with self.assertRaises(ValueError):
            retriever.rank_embeddings([1, 0], [[1, 0, 0]], [{"chunk_id": "x"}])

    def test_old_keyword_still_works(self):
        (self.root / "notes.md").write_text("工具超时允许有限重试。", encoding="utf-8")
        hits = retriever.retrieve("工具超时如何重试", method="keyword")
        self.assertTrue(hits)
        self.assertEqual(hits[0]["method"], "keyword")

if __name__ == "__main__":
    unittest.main()
