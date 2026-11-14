"""Day8：离线验证Hit@K判定与异常记录，不连接真实Chroma。"""
import unittest
from eval_day8 import evaluate, matches_gold

CASE = {
    "id": "Q01",
    "question": "索引更新后应先做什么？",
    "gold_source": "20_day8_eval.md",
    "gold_text": "索引更新后先运行固定问题回归检查。",
}


class Day8Tests(unittest.TestCase):
    def test_correct_source_and_text(self):
        self.assertTrue(matches_gold({
            "source": "20_day8_eval.md",
            "text": CASE["gold_text"],
        }, CASE))

    def test_same_source_wrong_text_is_not_hit(self):
        self.assertFalse(matches_gold({
            "source": "20_day8_eval.md",
            "text": "完全不相关的片段",
        }, CASE))

    def test_wrong_source_is_not_hit(self):
        self.assertFalse(matches_gold({
            "source": "other.md",
            "text": CASE["gold_text"],
        }, CASE))

    def test_top_k_hit(self):
        def fake_search(question, top_k):
            return [{"source": "other.md", "text": "无关"},
                    {"source": "20_day8_eval.md", "text": CASE["gold_text"]}][:top_k]
        self.assertEqual(evaluate([CASE], k=3, search_fn=fake_search)[0]["hit"], 1)
        self.assertEqual(evaluate([CASE], k=1, search_fn=fake_search)[0]["hit"], 0)

    def test_exception_is_recorded(self):
        def broken(question, top_k):
            raise RuntimeError("模拟索引失败")
        row = evaluate([CASE], search_fn=broken)[0]
        self.assertEqual(row["hit"], 0)
        self.assertIn("RuntimeError", row["error"])


if __name__ == "__main__":
    unittest.main()