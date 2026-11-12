"""Day5：交互式检索，输入“更新索引”即可同步知识库文件。"""
from openai import APIError
from retriever import retrieve
from vector_store import sync_index, search_index
from llm import answer

def main():
    print('DevDocs Agent Day5｜输入“更新索引”或“退出”。')
    print('第一次请先输入“更新索引”，随后再提问。')
    while True:
        try:
            question = input('\n你的问题：').strip()
        except (EOFError, KeyboardInterrupt):
            print('\n已结束。')
            return
        if question == '退出':
            return
        if question == '更新索引':
            try:
                report = sync_index()
                print('[索引同步]', report)
                print(report['added'], '个新块，', report['removed'], '个已删除，', report['unchanged'], '个未变。')
            except Exception as exc:
                print(f'[索引同步失败] {type(exc).__name__}: {exc}')
            continue
        if not question:
            print('请输入问题。')
            continue
        try:
            hits = search_index(question, top_k=3)
            print('[Chroma检索]')
            for h in hits:
                print(f" {h['chunk_id']} | source={h['source']} | cosine={h['score']} | {h['text'][:70]}")
            if not hits:
                print('索引为空；先输入“更新索引”。')
                continue
        except Exception as exc:
            print(f'[Chroma不可用] {type(exc).__name__}: {exc}')
            print('尝试关键词检索兜底。')
            hits = retrieve(question, top_k=3, method='keyword')
        if not hits:
            print('没有找到证据。')
            continue
        try:
            print('[回答]', answer(question, hits))
        except APIError as exc:
            print(f'[模型接口失败] {type(exc).__name__}: 请检查密钥、余额和模型名。')
        except (OSError, ValueError) as exc:
            print(f'[模型配置失败] {type(exc).__name__}: {exc}')

if __name__ == '__main__':
    main()