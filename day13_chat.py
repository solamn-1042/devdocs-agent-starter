"""Day13：为Day12 Agent增加可测试的多轮追问与历史裁剪。"""
from agent_day12 import run_agent

BUDGET = 1200  # 字符近似预算，不是精确token数
MAX_CALLS_PER_ROUND = 3  # 可一次读取检索返回的三个候选片段

def trim_history(history, budget=BUDGET):
    """从最新轮次向前保留完整问答；旧轮次超预算即舍弃。"""
    if budget < 0:
        raise ValueError("budget不能为负数")
    kept, used = [], 0
    for turn in reversed(history):
        size = len(turn["question"]) + len(turn["answer"])
        if used + size > budget:
            break
        kept.append(turn.copy())
        used += size
    return list(reversed(kept))

def rewrite_question(question, history):
    """仅在短追问含指代词时拼入最近用户问题，不引用旧模型回答。"""
    if not history or not any(x in question for x in ("它", "那", "这个", "上述", "继续")):
        return question
    return ("上一轮用户问题：" + history[-1]["question"][:240]
            + "\n本轮追问：" + question
            + "\n请根据上一轮主题理解指代；仍不清楚则澄清。")

def ask_turn(question, history, agent=None, budget=BUDGET):
    """使用已有run_agent执行，返回真实回答、轨迹和新历史。"""
    question = question.strip()
    if not question or len(question) > 500:
        raise ValueError("问题不能为空且不能超过500字符")
    limited = trim_history(history, budget)
    effective = rewrite_question(question, limited)
    if agent is None:
        result = run_agent(effective, max_calls_per_round=MAX_CALLS_PER_ROUND)
    else:
        result = agent(effective)
    answer = str(result.get("answer", ""))
    updated = trim_history(limited + [{
        "question": question, "answer": answer[:500]
    }], budget)
    return {"answer": answer, "trace": result.get("trace", []),
            "effective_question": effective, "history": updated,
            "history_chars": sum(len(x["question"]) + len(x["answer"]) for x in updated)}

def main():
    """连续自然语言问答；清空命令重置本次会话。"""
    history = []
    print("多轮助手已启动。输入“清空”重置，输入“退出”结束。")
    while True:
        try:
            question = input("你的问题：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question == "退出":
            break
        if question == "清空":
            history = []
            print("已清空会话历史。")
            continue
        if not question:
            print("请输入问题。")
            continue
        try:
            result = ask_turn(question, history)
            history = result["history"]
            print("实际提交问题：", result["effective_question"])
            for line in result["trace"]:
                print("[trace]", line)
            print("回答：", result["answer"])
            print("历史字符数：", result["history_chars"])
        except Exception as exc:
            print("本轮失败：", type(exc).__name__, "；历史未更新。")

if __name__ == "__main__":
    main()