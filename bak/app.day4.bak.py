"""Day 4：同一问题对比关键词检索和语义检索。"""
from openai import APIError
from retriever import retrieve
from llm import answer

def main() -> None:
    """输入自然语言问题，显示两种检索结果，再生成回答。"""
    print("DevDocs Agent Day 4；输入'退出'结束。")
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

        keyword_hits = retrieve(question, top_k=3, method="keyword")
        print("\n[关键词检索]")
        for hit in keyword_hits:
            print(f"  {hit['chunk_id']} | score={hit['score']} | {hit['text'][:60]}")

        # 模型首次下载可能需要联网；失败时只使用原关键词结果。
        try:
            semantic_hits = retrieve(question, top_k=3, method="semantic")
            print("\n[语义检索]")
            for hit in semantic_hits:
                print(f"  {hit['chunk_id']} | cosine={hit['score']} | {hit['text'][:60]}")
            selected = semantic_hits
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            print(f"[语义检索不可用] {type(exc).__name__}: {exc}")
            selected = keyword_hits

        # 只有配置了真实API后才可能获得模型回答。
        try:
            print("\n[回答]", answer(question, selected))
        except APIError as exc:
            print(f"[模型接口失败] {type(exc).__name__}: 检查密钥、余额、模型名")

if __name__ == "__main__":
    main()