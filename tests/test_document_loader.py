"""Day 2：使用临时文件夹测试，不修改真实 knowledge/。"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import retriever


class TestDocumentLoader(unittest.TestCase):
    def setUp(self):
        # 每项测试都有独立知识库，避免污染用户原有文件。
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.patcher = patch.object(retriever, "DATA_DIR", self.root)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_load_md_and_txt(self):
        (self.root / "note.md").write_text("文档检索规范", encoding="utf-8")
        (self.root / "rules.txt").write_text("接口限流\n重试上限", encoding="utf-8")
        docs = retriever.load_documents()
        self.assertEqual({d["source"] for d in docs}, {"note.md", "rules.txt"})

    def test_metadata(self):
        (self.root / "rules.txt").write_text("接口限流\n重试上限", encoding="utf-8")
        doc = retriever.load_documents()[0]
        self.assertEqual(doc["file_type"], ".txt")
        self.assertEqual(doc["line_count"], 2)
        self.assertEqual(doc["char_count"], len("接口限流\n重试上限"))

    def test_ignore_other_formats_and_empty(self):
        (self.root / "fake.pdf").write_text("接口限流", encoding="utf-8")
        (self.root / "empty.txt").write_text("  ", encoding="utf-8")
        self.assertEqual(retriever.load_documents(), [])

    def test_retrieve_txt_source(self):
        (self.root / "api_rules.txt").write_text(
            "接口限流达到阈值后返回429，并记录请求ID。", encoding="utf-8"
        )
        hits = retriever.retrieve("接口限流达到阈值怎么办？")
        self.assertEqual(hits[0]["source"], "api_rules.txt")
        self.assertGreaterEqual(hits[0]["score"], 2)
        self.assertEqual(hits[0]["file_type"], ".txt")

    def test_top_k(self):
        (self.root / "a.md").write_text("接口限流策略", encoding="utf-8")
        (self.root / "b.txt").write_text("接口限流日志", encoding="utf-8")
        self.assertEqual(len(retriever.retrieve("接口限流", top_k=1)), 1)
        self.assertEqual(retriever.retrieve("接口限流", top_k=0), [])


if __name__ == "__main__":
    unittest.main()