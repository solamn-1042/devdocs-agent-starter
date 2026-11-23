"""Day14：离线测试，不访问真实API。"""
import json
import tempfile
import time
import unittest
from pathlib import Path
from day14_trace import safe_turn, classify

def fake_agent(question):
    """模拟Day12返回值。"""
    return {"answer": "请重新同步Chroma索引", "trace": ["mock_only"]}

class Day14Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "trace.jsonl"
        self.recent = set()

    def tearDown(self):
        self.temp.cleanup()

    def ask(self, question, runner=fake_agent, timeout=1):
        return safe_turn(question, [], runner=runner, recent=self.recent,
                         timeout=timeout, trace_path=self.path)

    def test_success(self):
        self.assertEqual(self.ask("索引更新后怎么办")["status"], "ok")

    def test_trace(self):
        self.ask("索引更新后怎么办")
        item = json.loads(self.path.read_text(encoding="utf-8").splitlines()[0])
        self.assertIn("trace_id", item)
        self.assertNotIn("索引更新后", str(item))

    def test_duplicate(self):
        self.ask("相同问题")
        self.assertEqual(self.ask("相同问题")["status"], "duplicate")

    def test_invalid(self):
        self.assertEqual(self.ask("  ")["trace"][0]["error"], "invalid_input")

    def test_timeout(self):
        def slow(question):
            time.sleep(0.1)
            return fake_agent(question)
        self.assertEqual(self.ask("慢请求", runner=slow, timeout=0.01)
                         ["trace"][0]["error"], "timeout")

    def test_failure_keeps_history(self):
        old = [{"question": "原问题", "answer": "原回答"}]
        def broken(question):
            raise RuntimeError("模拟异常")
        result = safe_turn("新问题", old, runner=broken, recent=self.recent,
                           trace_path=self.path)
        self.assertEqual(result["history"], old)
        self.assertEqual(classify(ValueError("bad")), "invalid_input")

if __name__ == "__main__":
    unittest.main()