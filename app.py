"""Day 3：终端展示命中的 Chunk ID、行号和真实来源。"""
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
        print(f"[检索] 找到 {len(hits)} 个文档片段")
        for hit in hits:
            print(
                f" - {hit['chunk_id']} | {hit['source']} "
                f"| 第{hit['line_start']}-{hit['line_end']}行 "
                f"| 类型 {hit['file_type']} | 分数 {hit['score']}"
            )
            print(f"   摘要：{hit['text'][:70]}")
        try:
            print("[回答]", answer(question, hits))
        except APIError as exc:
            print(f"[模型接口失败] {type(exc).__name__}：检查密钥、余额和模型名。")


if __name__ == "__main__":
    main()