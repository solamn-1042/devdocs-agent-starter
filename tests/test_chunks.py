"""Day 3：分块和来源追踪离线测试，不调用模型 API。"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import retriever


class TestChunks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.patcher = patch.object(retriever, "DATA_DIR", self.root)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_blank_line_splits_paragraphs(self):
        (self.root / "rules.md").write_text(
            "接口限流返回429。\n\n工具超时允许有限重试。", encoding="utf-8"
        )
        chunks = retriever.load_chunks()
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["chunk_id"], "rules.md#chunk_001")
        self.assertEqual(chunks[1]["line_start"], 3)

    def test_long_paragraph_splits(self):
        (self.root / "long.txt").write_text("检索" * 20, encoding="utf-8")
        chunks = retriever.load_chunks(max_chars=15)
        self.assertEqual(len(chunks), 3)
        self.assertTrue(all(len(c["text"]) <= 15 for c in chunks))

    def test_retrieve_chunk_and_source(self):
        (self.root / "rules.md").write_text(
            "用户登录注意事项。\n\n工具超时允许有限重试。", encoding="utf-8"
        )
        hits = retriever.retrieve("工具超时后如何重试？")
        self.assertEqual(hits[0]["chunk_id"], "rules.md#chunk_002")
        self.assertEqual(hits[0]["line_start"], 3)
        self.assertEqual(hits[0]["source"], "rules.md")

    def test_empty_and_invalid_limit(self):
        (self.root / "empty.md").write_text("  ", encoding="utf-8")
        self.assertEqual(retriever.load_chunks(), [])
        with self.assertRaises(ValueError):
            retriever.split_document({"text": "示例", "source": "x.md"}, 0)

    def test_top_k_and_stable_id(self):
        (self.root / "a.txt").write_text(
            "接口限流日志。\n\n接口限流重试策略。", encoding="utf-8"
        )
        first = retriever.retrieve("接口限流", top_k=1)
        second = retriever.retrieve("接口限流", top_k=1)
        self.assertEqual(len(first), 1)
        self.assertEqual(first[0]["chunk_id"], second[0]["chunk_id"])


if __name__ == "__main__":
    unittest.main()