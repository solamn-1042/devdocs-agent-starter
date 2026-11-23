"""Day14：基于Day13的失败分类和JSONL Trace。"""
import json
import time
import uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from day13_chat import ask_turn

TRACE_PATH = Path(__file__).resolve().parent / "logs" / "day14_trace.jsonl"
TIMEOUT = 35

def classify(exc):
    """把异常转换为稳定类别，避免输出敏感请求。"""
    name = type(exc).__name__.lower()
    if "timeout" in name:
        return "timeout"
    if "rate" in name or "429" in str(exc):
        return "rate_limited"
    if "authentication" in name:
        return "auth_failed"
    if isinstance(exc, ValueError):
        return "invalid_input"
    return "internal_error"

def save_trace(item, path=TRACE_PATH):
    """每次追加一条JSON记录，首次自动创建logs目录。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

def call_limited(fn, seconds):
    """限制等待时间；底层线程超时后可能仍在运行。"""
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(fn)
    try:
        return future.result(timeout=seconds)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

def safe_turn(question, history, runner=None, recent=None,
              timeout=TIMEOUT, trace_path=TRACE_PATH):
    """调用Day13；成功才更新历史，重复问题不再执行。"""
    started = time.perf_counter()
    recent = set() if recent is None else recent
    question = question.strip()
    status, error, result = "ok", "", None
    if not question or len(question) > 500:
        status, error = "failed", "invalid_input"
    elif question in recent:
        status, error = "duplicate", "duplicate_question"
    else:
        try:
            result = call_limited(
                lambda: ask_turn(question, history, agent=runner), timeout)
            if not isinstance(result, dict) or "answer" not in result:
                raise ValueError("bad_result")
            if "config_missing" in result.get("trace", []):
                status, error = "failed", "config_missing"
            else:
                recent.add(question)
        except FutureTimeout:
            status, error = "failed", "timeout"
        except Exception as exc:
            status, error = "failed", classify(exc)
    trace = {"trace_id": uuid.uuid4().hex[:12], "status": status,
             "error": error, "elapsed_ms": round(
                 (time.perf_counter() - started) * 1000, 2),
             "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    save_trace(trace, trace_path)
    if status != "ok":
        return {"status": status, "answer": "本轮未完成：" + error,
                "history": history, "trace": [trace]}
    return {"status": "ok", "answer": result["answer"],
            "history": result["history"],
            "trace": result.get("trace", []) + [trace]}

def main():
    """命令行自然语言输入，清空可重置会话和去重记录。"""
    history, recent = [], set()
    print("Day14启动：输入 清空 或 退出。")
    while True:
        try:
            question = input("你的问题：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question == "退出":
            break
        if question == "清空":
            history, recent = [], set()
            print("会话已清空")
            continue
        result = safe_turn(question, history, recent=recent)
        history = result["history"]
        print("状态：", result["status"])
        print("回答：", result["answer"])
        for line in result["trace"]:
            print("[trace]", line)
        print("日志：", TRACE_PATH)

if __name__ == "__main__":
    main()