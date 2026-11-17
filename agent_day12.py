"""Day12：在已有Day10检索上运行受控的LLM工具调用图。"""
import json
import os
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from openai import OpenAI
from langgraph.graph import StateGraph, START, END
from rerank_day10 import safe_search

# 模型只看到工具说明；真正权限由dispatch执行。
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "从DevDocs本地知识库检索问题证据；先检索再按需要读取片段",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "用户想查询的研发文档问题"}
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_source",
            "description": "读取本次检索命中片段的完整文本，须提供已返回的chunk_id",
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk_id": {
                        "type": "string",
                        "description": "search_docs返回的Chunk ID",
                    }
                },
                "required": ["chunk_id"],
                "additionalProperties": False,
            },
        },
    },
]


class AgentState(TypedDict):
    question: str
    answer: str
    trace: list[str]


def dispatch(name: str, raw_args: str, cache: dict) -> dict:
    """校验工具名、JSON、字段、长度，再执行已批准的业务函数。"""
    if name not in {"search_docs", "read_source"}:
        return {"error": "tool_not_allowed"}
    try:
        args = json.loads(raw_args)
    except (ValueError, TypeError):
        return {"error": "invalid_json"}
    key = "query" if name == "search_docs" else "chunk_id"
    if not isinstance(args, dict) or set(args) != {key}:
        return {"error": "invalid_fields"}
    value = args[key]
    if not isinstance(value, str) or not value.strip() or len(value) > 200:
        return {"error": "invalid_value"}
    value = value.strip()
    if name == "read_source":
        hit = cache.get(value)
        if hit is None:
            return {"error": "chunk_not_in_current_search"}
        return {
            "source": hit.get("source", ""),
            "chunk_id": value,
            "text": str(hit.get("text", ""))[:3000],
        }
    try:
        result = safe_search(value, top_k=3)
    except Exception:
        return {"error": "search_exception"}
    cache.clear()  # 旧查询的Chunk不能被下一次检索继续读取
    rows = []
    for hit in result.get("hits", []):
        cid = str(hit.get("chunk_id", ""))
        if not cid:
            continue
        cache[cid] = hit
        rows.append(
            {
                "source": str(hit.get("source", "")),
                "chunk_id": cid,
                "preview": str(hit.get("text", ""))[:240],
            }
        )
    return {"status": result.get("status", "unknown"), "hits": rows}


def run_agent(question: str, client=None, model=None) -> dict:
    """最多执行两轮、每轮最多一个工具调用；工具结果交回模型。"""
    load_dotenv()
    model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    if client is None:
        key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if not key:
            return {
                "answer": "缺少DEEPSEEK_API_KEY，请检查项目根目录.env。",
                "trace": ["config_missing"],
            }
        client = OpenAI(
            api_key=key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            timeout=30,
            max_retries=0,
        )
    messages = [
        {
            "role": "system",
            "content": (
                "你是DevDocs文档助手。需要文档事实时先调用search_docs，"
                "可再用read_source读取已检索的chunk_id。"
                "仅根据返回的证据回答，标明source和chunk_id；证据不足要直说。"
            ),
        },
        {"role": "user", "content": question},
    ]
    cache = {}
    trace = []
    for step in range(2):
        response = client.chat.completions.create(
            model=model, messages=messages, tools=TOOLS, tool_choice="auto"
        )
        msg = response.choices[0].message
        calls = msg.tool_calls or []
        if not calls:
            return {"answer": msg.content or "模型未返回内容", "trace": trace}
        if len(calls) != 1:
            return {"answer": "已停止：每轮最多允许一个工具调用。", "trace": trace + ["too_many_calls"]}
        call = calls[0]
        # 记录工具名称和错误类别，不打印密钥或完整参数。
        result = dispatch(call.function.name, call.function.arguments, cache)
        trace.append(
            f"step={step+1} tool={call.function.name} "
            f"status={result.get('error', result.get('status', 'ok'))}"
        )
        messages.append(msg.model_dump(exclude_none=True))
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result, ensure_ascii=False),
            }
        )
    # 两轮之后关闭工具入口，要求模型根据已有证据总结。
    final = client.chat.completions.create(model=model, messages=messages)
    return {"answer": final.choices[0].message.content or "无回答", "trace": trace}


def agent_node(state: AgentState) -> dict:
    """LangGraph节点调用真实Agent，沿用Day11的StateGraph思路。"""
    return run_agent(state["question"])


def build_graph():
    """编译输入→Agent工具决策→输出的LangGraph图。"""
    builder = StateGraph(AgentState)
    builder.add_node("agent_tools", agent_node)
    builder.add_edge(START, "agent_tools")
    builder.add_edge("agent_tools", END)
    return builder.compile()


def main():
    """在PowerShell直接输入自然语言，展示工具轨迹和回答。"""
    print("DevDocs Agent工具调用已启动；输入“退出”结束。")
    graph = build_graph()
    while True:
        try:
            question = input("你的问题：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question == "退出":
            break
        if not question:
            print("请输入问题。")
            continue
        try:
            result = graph.invoke({"question": question, "answer": "", "trace": []})
            for line in result["trace"]:
                print("[trace]", line)
            print("回答：", result["answer"])
        except Exception as exc:
            print("请求失败：", type(exc).__name__, "（检查网络、余额与模型名）")


if __name__ == "__main__":
    main()
