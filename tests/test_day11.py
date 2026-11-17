"""Day11：通过替换节点验证状态流转，不访问Chroma或API。"""
import unittest
from langgraph.graph import StateGraph, START, END
from langgraph_day11 import DocState, ask, build_graph


class Day11Tests(unittest.TestCase):
    def test_graph_compiles(self):
        self.assertIsNotNone(build_graph())

    def test_state_and_evidence_flow(self):
        # 注入固定节点，验证LangGraph执行顺序与字段传递。
        def fake_retrieve(state):
            return {"hits": [{"source": "demo.md", "chunk_id": "c1",
                              "text": "索引更新后应重新同步。"}],
                    "search_status": "reranked"}

        def fake_answer(state):
            return {"response": "依据" + state["hits"][0]["source"] + "回答"}

        builder = StateGraph(DocState)
        builder.add_node("retrieve", fake_retrieve)
        builder.add_node("answer", fake_answer)
        builder.add_edge(START, "retrieve")
        builder.add_edge("retrieve", "answer")
        builder.add_edge("answer", END)
        result = ask("索引更新后怎么办？", builder.compile())
        self.assertEqual(result["search_status"], "reranked")
        self.assertEqual(result["hits"][0]["chunk_id"], "c1")
        self.assertEqual(result["response"], "依据demo.md回答")

    def test_empty_evidence_branch(self):
        # 真正answer_node在空证据时应直接拒答。
        from langgraph_day11 import answer_node
        state = {"question": "火星天气", "hits": [],
                 "search_status": "empty", "response": ""}
        self.assertIn("未找到", answer_node(state)["response"])


if __name__ == "__main__":
    unittest.main()