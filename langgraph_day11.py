"""Day11：把Day10真实检索和已有LLM回答串成LangGraph状态图。"""
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from rerank_day10 import safe_search
from llm import answer


class DocState(TypedDict):
    """节点间共享的数据：问题、证据、状态、回答。"""
    question: str
    hits: list[dict]
    search_status: str
    response: str


def retrieve_node(state: DocState) -> dict:
    """调用Day10真实检索；只更新检索相关字段。"""
    result = safe_search(state["question"], top_k=3)
    return {
        "hits": result["hits"],
        "search_status": result["status"],
    }


def answer_node(state: DocState) -> dict:
    """复用Day1的llm.answer；无证据时避免请求模型。"""
    hits = state["hits"]
    if not hits:
        return {"response": "知识库未找到可用证据，请补充文档。"}
    try:
        return {"response": answer(state["question"], hits)}
    except Exception as exc:
        # 只记录异常类型，不输出密钥或请求正文。
        return {"response": f"模型调用失败：{type(exc).__name__}，请检查配置。"}


def build_graph():
    """将已有业务函数接成固定的检索→回答流程。"""
    builder = StateGraph(DocState)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("answer", answer_node)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "answer")
    builder.add_edge("answer", END)
    return builder.compile()


graph = build_graph()


def ask(question: str, workflow=None) -> DocState:
    """供命令行、后续Agent/API复用的统一入口。"""
    runner = graph if workflow is None else workflow
    return runner.invoke({
        "question": question,
        "hits": [],
        "search_status": "pending",
        "response": "",
    })


def main():
    """用户直接输入自然语言，显示来源与回答。"""
    print("DevDocs LangGraph 已启动；输入“退出”结束。")
    while True:
        try:
            question = input("你的问题：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question == "退出":
            break
        if not question:
            print("请先输入问题。")
            continue
        result = ask(question)
        print("检索状态：", result["search_status"])
        for hit in result["hits"]:
            print("来源：", hit.get("source"), "Chunk：", hit.get("chunk_id"))
        print("回答：", result["response"])


if __name__ == "__main__":
    main()