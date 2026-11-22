"""Day15：复用Day14安全问答，提供本地HTTP API。"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from day14_trace import safe_turn

app = FastAPI(title="DevDocs Agent API", version="0.15.0")

class Turn(BaseModel):
    """一轮历史：字段名与Day13保持一致。"""
    question: str
    answer: str

class ChatRequest(BaseModel):
    """请求包含问题和可选历史；每次请求自行携带上下文。"""
    question: str = Field(min_length=1, max_length=500)
    history: list[Turn] = Field(default_factory=list, max_length=20)

@app.get("/health")
def health():
    """只确认HTTP进程可用，不声称Chroma或模型正常。"""
    return {"status": "ok", "service": "devdocs-agent"}

@app.post("/chat")
def chat(request: ChatRequest):
    """调用既有Day14包装，统一返回成功或失败状态。"""
    history = [turn.model_dump() for turn in request.history]
    result = safe_turn(request.question, history)
    if result["status"] != "ok":
        # 日志由Day14记录；HTTP返回稳定错误类型。
        error = result["trace"][-1].get("error", "internal_error")
        code = 400 if error in {"invalid_input", "duplicate_question"} else 503
        raise HTTPException(status_code=code, detail={"error": error})
    return {"status": "ok", "answer": result["answer"],
            "history": result["history"], "trace": result["trace"]}