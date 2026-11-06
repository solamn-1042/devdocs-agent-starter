"""终端交互入口：看得到检索来源、分数和模型回答。"""
from openai import APIError
from retriever import retrieve
from llm import answer

def main() -> None:
    print("研发文档助手已启动：输入问题；输入'退出'结束。")
    while True:
        try:
            question = input("\n你的问题：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已结束。")
            return
        if question == "退出":
            return
        if not question:
            print("请输入问题。")
            continue
        hits = retrieve(question)
        print(f"[检索] 找到 {len(hits)} 条证据")
        for hit in hits:
            print(f"  - {hit['source']} | 关键词匹配分数 {hit['score']}")
        try:
            print("[回答]", answer(question, hits))
        except APIError as exc:
            # 不打印请求对象、密钥等隐私信息；排查时再检查完整Trace。
            print(f"[模型接口失败] {type(exc).__name__}：检查密钥、余额和模型名。")

if __name__ == "__main__":
    main()
