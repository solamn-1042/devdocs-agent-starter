"""Day13：完全离线的多轮上下文测试。"""
import unittest
from unittest.mock import patch
from day13_chat import trim_history, rewrite_question, ask_turn

def fake_agent(question):
    """模拟Day12返回结构，不访问网络和Chroma。"""
    return {"answer": "模拟回答", "trace": ["mock_only"]}

class Day13Tests(unittest.TestCase):
    def test_first_turn(self):
        result = ask_turn("Chroma怎么同步？", [], agent=fake_agent)
        self.assertEqual(result["effective_question"], "Chroma怎么同步？")

    def test_follow_up(self):
        first = ask_turn("Chroma怎么同步？", [], agent=fake_agent)
        second = ask_turn("那它失败时怎么办？", first["history"], agent=fake_agent)
        self.assertIn("Chroma怎么同步？", second["effective_question"])

    def test_default_agent_enables_batch_reads_for_follow_up(self):
        history = [{"question": "Chroma索引更新后要做什么？", "answer": "重新同步。"}]
        with patch("day13_chat.run_agent", return_value=fake_agent("")) as runner:
            result = ask_turn("那它同步失败怎么办？", history)
        self.assertIn(history[0]["question"], result["effective_question"])
        runner.assert_called_once_with(result["effective_question"], max_calls_per_round=3)

    def test_new_topic(self):
        history = [{"question": "索引是什么", "answer": "示例"}]
        self.assertEqual(rewrite_question("解释BM25原理", history), "解释BM25原理")

    def test_budget(self):
        history = [{"question": "旧问题", "answer": "A" * 80},
                   {"question": "新问题", "answer": "B" * 10}]
        self.assertEqual(len(trim_history(history, 30)), 1)

    def test_empty(self):
        with self.assertRaises(ValueError):
            ask_turn("", [], agent=fake_agent)

    def test_original_unchanged(self):
        history = [{"question": "原问题", "answer": "原回答"}]
        ask_turn("新问题", history, agent=fake_agent)
        self.assertEqual(len(history), 1)

if __name__ == "__main__":
    unittest.main()