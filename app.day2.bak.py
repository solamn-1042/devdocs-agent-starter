"""终端入口：展示文件名、类型、行数、匹配分数和模型回答。"""
from openai import APIError
from retriever import retrieve
from llm import answer


def main() -> None:
    print("DevDocs Agent 已启动；输入问题，输入'退出'结束。")
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
            print(
                f" - {hit['source']} | 类型 {hit['file_type']} "
                f"| {hit['line_count']} 行 "
                f"| 关键词匹配分数 {hit['score']}"
            )
        try:
            print("[回答]", answer(question, hits))
        except APIError as exc:
            # 不输出 API Key 或请求原文。
            print(
                f"[模型接口失败] {type(exc).__name__}："
                "检查密钥、余额和模型名。"
            )


if __name__ == "__main__":
    main()