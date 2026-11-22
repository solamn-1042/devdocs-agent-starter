"""Day16最终审计：检查固定样本、工程文件与回归退出码。"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GOLD = ROOT / "evals" / "day8_cases.json"
REPORT = ROOT / "reports" / "day16_audit.json"
FILES = [
    "app.py", "retriever.py", "llm.py", "eval_day8.py",
    "agent_day12.py", "day13_chat.py", "day14_trace.py",
    "api_day15.py", "tests/test_day15.py", "requirements.txt",
]

def read_cases():
    """读取原有Day8 Gold，兼容常见的数组容器。"""
    data = json.loads(GOLD.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("cases", "questions", "data", "items"):
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError("无法识别Gold结构，请对照Day8原文件")

def execute(args):
    """运行已有测试，保留实际退出码、耗时和日志。"""
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            args, cwd=ROOT, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=180
        )
        return {"command": args, "exit_code": proc.returncode,
                "elapsed_ms": round((time.perf_counter()-started)*1000, 2),
                "log": (proc.stdout + proc.stderr)[-6000:]}
    except Exception as exc:
        return {"command": args, "exit_code": -1,
                "log": type(exc).__name__ + ": " + str(exc)}

def main():
    """输出报告；任何缺失或回归失败均返回非零状态。"""
    if not GOLD.is_file():
        raise FileNotFoundError("缺少evals/day8_cases.json")
    cases = read_cases()
    checks = [{"name": "file:" + name,
               "passed": (ROOT / name).is_file()} for name in FILES]
    checks += [{"name": "gold:" + str(i+1),
                "passed": i < len(cases) and isinstance(cases[i], dict)}
               for i in range(20)]
    commands = [
        [sys.executable, "-m", "unittest", "discover",
         "-s", "tests", "-p", "test_day15.py", "-v"],
        [sys.executable, "eval_day8.py"],
    ]
    results = [execute(command) for command in commands]
    report = {"gold_count": len(cases), "checks": checks,
              "passed": sum(x["passed"] for x in checks),
              "runs": results,
              "warning": "Gold存在性不是Hit@K；须查看Day8实际输出"}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print("固定检查：", report["passed"], "/", len(checks))
    for result in results:
        print("退出码：", result["exit_code"])
        print(result["log"][-1000:])
    print("报告路径：", REPORT)
    if len(cases) != 20 or report["passed"] != 30:
        sys.exit(1)
    if any(result["exit_code"] != 0 for result in results):
        sys.exit(1)

if __name__ == "__main__":
    main()