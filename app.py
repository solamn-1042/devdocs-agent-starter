"""Day6：Chroma 检索后执行证据门禁，回答必须通过引用校验。"""
from openai import APIError
from retriever import retrieve
from vector_store import sync_index, search_index
from evidence import assess_evidence, REFUSAL_MESSAGE
from llm import answer


def main():
    """支持更新索引、自然语言问答、来源显示和证据不足拒答。"""
    print("DevDocs Agent Day6｜输入“更新索引”或“退出”。")
    print("首次运行或修改知识文档后，请先输入“更新索引”。")

    while True:
        try:
            question = input("\n你的问题：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已结束。")
            return

        if question == "退出":
            return

        if question == "更新索引":
            try:
                report = sync_index()
                print("[索引同步]", report)
            except Exception as exc:
                print(f"[索引同步失败] {type(exc).__name__}: {exc}")
            continue

        if not question:
            print("请输入问题。")
            continue

        try:
            hits = search_index(question, top_k=3)
            search_name = "Chroma"
        except Exception as exc:
            print(f"[Chroma不可用] {type(exc).__name__}: {exc}")
            print("[降级] 尝试原关键词检索。")
            hits = retrieve(question, top_k=3, method="keyword")
            search_name = "关键词"

        print(f"\n[{search_name}检索]")
        for hit in hits:
            print(
                f" {hit.get('chunk_id')} | source={hit.get('source')} "
                f"| score={hit.get('score')} "
                f"| 行={hit.get('line_start', '?')}-{hit.get('line_end', '?')}"
            )
            print(f"   {hit.get('text', '')[:100]}")

        status = assess_evidence(hits)
        print(
            f"[证据门禁] {'通过' if status['ok'] else '拒绝'}"
            f"｜{status['reason']}"
        )

        if not status["ok"]:
            # 门禁拒绝时不调用模型，直接给出确定性拒答。
            print("[回答]", REFUSAL_MESSAGE)
            continue

        try:
            print("[回答]", answer(question, hits))
        except APIError as exc:
            print(
                f"[模型接口失败] {type(exc).__name__}："
                "请检查密钥、余额、模型名和服务状态。"
            )
        except (OSError, ValueError) as exc:
            print(f"[模型配置失败] {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()