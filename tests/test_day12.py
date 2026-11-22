"""Day12：离线测试工具执行边界，不发模型API请求。"""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from agent_day12 import dispatch, build_graph, run_agent, TOOLS

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

    def run_tool_batch(self, size, *, raw_tool_content=None,
                       final_content="索引更新后应重新同步 Chroma。", **options):
        """模拟检索后，模型一次请求读取多个片段。"""
        def call(call_id, name, args):
            return SimpleNamespace(id=call_id, function=SimpleNamespace(
                name=name, arguments=args))

        def response(calls, content=None):
            message = Mock(tool_calls=calls, content=content)
            message.model_dump.return_value = {
                "role": "assistant", "tool_calls": [{"id": c.id} for c in calls]
            }
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

        client = Mock()
        create = client.chat.completions.create
        create.side_effect = [
            response([call("search", "search_docs", '{"query":"Chroma索引"}')]),
            response([call(f"read-{i}", "read_source", '{"chunk_id":"c001"}')
                      for i in range(size)], raw_tool_content),
            response([], final_content),
        ]
        with patch("agent_day12.safe_search", side_effect=fake_search):
            result = run_agent("Chroma索引更新后要做什么？", client=client,
                               model="test", **options)
        return result, create

    def test_day12_keeps_single_call_baseline(self):
        result, create = self.run_tool_batch(2)
        self.assertEqual(result["trace"][-1], "too_many_calls")
        self.assertEqual(create.call_count, 2)

    def test_batched_calls_each_return_a_result(self):
        result, create = self.run_tool_batch(3, max_calls_per_round=3)
        self.assertIn("重新同步 Chroma", result["answer"])
        self.assertEqual(result["trace"].count("step=2 tool=read_source status=ok"), 3)
        self.assertEqual(create.call_count, 3)
        self.assertNotIn("tools", create.call_args.kwargs)
        messages = create.call_args_list[1].kwargs["messages"]
        self.assertEqual([m["tool_call_id"] for m in messages if m["role"] == "tool"],
                         ["search", "read-0", "read-1", "read-2"])

    def test_oversized_batch_falls_back_to_search_evidence(self):
        result, create = self.run_tool_batch(4, max_calls_per_round=3)
        self.assertIn("重新同步 Chroma", result["answer"])
        self.assertEqual(result["trace"], ["step=1 tool=search_docs status=reranked",
                                           "too_many_calls count=4 limit=3"])
        self.assertEqual(create.call_count, 3)
        self.assertNotIn("tools", create.call_args.kwargs)
        messages = create.call_args.kwargs["messages"]
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        payload = json.loads(messages[-1]["content"])
        self.assertEqual(payload["question"], "Chroma索引更新后要做什么？")
        self.assertEqual(payload["evidence"], [{
            "source": "knowledge/demo.md", "chunk_id": "c001",
            "text": "索引更新后应重新同步Chroma。",
        }])

    def test_raw_tool_markup_in_summary_shows_evidence_instead(self):
        result, _ = self.run_tool_batch(4, max_calls_per_round=3,
                                       final_content='<｜｜DSML｜｜ invoke name="read_source">')
        self.assertNotIn("DSML", result["answer"])
        self.assertIn("knowledge/demo.md / c001", result["answer"])
        self.assertEqual(result["trace"][-1], "evidence_excerpt_fallback")

    def test_raw_tool_markup_before_summary_is_not_returned(self):
        result, create = self.run_tool_batch(0, max_calls_per_round=3,
                                            raw_tool_content="<｜｜DSML｜｜ calls>")
        self.assertIn("重新同步 Chroma", result["answer"])
        self.assertEqual(result["trace"][-1], "raw_tool_markup")
        self.assertNotIn("DSML", create.call_args.kwargs["messages"][-1]["content"])

if __name__ == "__main__":
    unittest.main()