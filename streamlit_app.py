"""Day7：DevDocs Agent 的 Streamlit 本地网页入口。"""
from pathlib import Path

import streamlit as st
from openai import APIError

from retriever import retrieve
from vector_store import get_collection, sync_index, search_index
from evidence import assess_evidence, REFUSAL_MESSAGE
from llm import answer
from upload_utils import safe_upload_name, decode_uploaded_text


# 所有文件路径都相对于当前项目，不依赖启动命令所在的目录。
PROJECT_DIR = Path(__file__).resolve().parent
KNOWLEDGE_DIR = PROJECT_DIR / "knowledge"


@st.cache_resource
def get_cached_collection():
    """在页面多次执行之间复用本地 Chroma Collection。"""
    return get_collection()


def save_uploaded_files(uploaded_files, overwrite_same_name: bool = False) -> dict:
    """校验并保存上传文件；默认不覆盖知识库中已有的同名文件。"""
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "saved": [],
        "skipped": [],
        "errors": [],
    }

    for uploaded_file in uploaded_files or []:
        original_name = getattr(uploaded_file, "name", "unknown")

        try:
            # 文件名不能直接作为磁盘路径使用。
            safe_name = safe_upload_name(original_name)

            # getvalue() 每次都返回完整字节，不受文件读取指针位置影响。
            content = decode_uploaded_text(uploaded_file.getvalue())

            target = KNOWLEDGE_DIR / safe_name

            # 默认保护原文件；只有用户明确勾选覆盖时才替换。
            if target.exists() and not overwrite_same_name:
                report["skipped"].append(
                    f"{safe_name}（同名文件已存在，未覆盖）"
                )
                continue

            target.write_text(content, encoding="utf-8")
            report["saved"].append(safe_name)

        except (ValueError, OSError) as exc:
            # 单个文件失败不影响其他上传文件。
            report["errors"].append(
                f"{original_name}：{type(exc).__name__}：{exc}"
            )

    return report


def run_question(question: str) -> dict:
    """复用 Day6 的检索、证据门禁和模型回答流程。"""
    fallback_note = ""
    search_name = "Chroma 向量检索"

    try:
        # 缓存的 Collection 与 Day5 的本地 chroma_db/ 指向同一索引。
        collection = get_cached_collection()
        hits = search_index(
            question,
            top_k=3,
            collection=collection,
        )
    except Exception as chroma_exc:
        # Chroma 异常时沿用 Day6 的关键词降级路线。
        fallback_note = (
            f"Chroma 检索异常：{type(chroma_exc).__name__}；"
            "本次尝试使用关键词检索。"
        )
        search_name = "关键词降级"

        try:
            hits = retrieve(
                question,
                top_k=3,
                method="keyword",
            )
        except Exception as fallback_exc:
            return {
                "error": (
                    "Chroma 与关键词检索均失败："
                    f"{type(fallback_exc).__name__}"
                )
            }

    # 证据门禁和引用校验继续使用 Day6 的实现，不在页面中复制规则。
    evidence_status = assess_evidence(hits)

    if not evidence_status["ok"]:
        # 证据不足时不调用模型，页面显示确定性拒答。
        answer_text = REFUSAL_MESSAGE
    else:
        try:
            answer_text = answer(question, hits)
        except APIError as exc:
            answer_text = (
                f"模型接口失败：{type(exc).__name__}。"
                "请检查 .env 中的密钥、模型名、余额和服务状态。"
            )
        except Exception as exc:
            answer_text = (
                f"回答阶段发生异常：{type(exc).__name__}。"
                "本次检索来源仍保留在页面中，便于排查。"
            )

    return {
        "question": question,
        "search_name": search_name,
        "fallback_note": fallback_note,
        "hits": hits,
        "gate_ok": evidence_status["ok"],
        "gate_reason": evidence_status["reason"],
        "answer": answer_text,
        "citation_blocked": "未通过来源引用校验" in answer_text,
    }


def main():
    """绘制知识库管理、自然语言提问和来源结果三个区域。"""
    st.set_page_config(
        page_title="DevDocs Agent",
        page_icon="📚",
        layout="wide",
    )

    # Session State 保存当前浏览器会话最近五次问答。
    if "qa_history" not in st.session_state:
        st.session_state["qa_history"] = []

    st.title("DevDocs Agent")
    st.caption(
        "研发文档智能知识库助手｜本地 Streamlit 界面｜"
        "上传、同步、检索、证据门禁与引用回答"
    )

    # 左侧：上传文档与查看当前知识库文件。
    with st.sidebar:
        st.header("知识库管理")
        st.caption("只接受 UTF-8 编码的 Markdown/TXT，单文件最多 2MB。")

        with st.form("upload_form"):
            uploaded_files = st.file_uploader(
                "选择知识库文件",
                type=["md", "txt"],
                accept_multiple_files=True,
                max_upload_size=2,
                help="可以一次选择多个 .md 或 .txt 文件。",
            )

            overwrite_same_name = st.checkbox(
                "允许覆盖 knowledge/ 中的同名文件",
                value=False,
                help="默认不覆盖，避免误删已有知识文档。",
            )

            upload_submitted = st.form_submit_button(
                "保存文件并同步索引",
                use_container_width=True,
            )

        if upload_submitted:
            file_report = save_uploaded_files(
                uploaded_files,
                overwrite_same_name=overwrite_same_name,
            )

            if file_report["saved"]:
                st.success("已保存：" + "、".join(file_report["saved"]))

            if file_report["skipped"]:
                for item in file_report["skipped"]:
                    st.info(item)

            if file_report["errors"]:
                for item in file_report["errors"]:
                    st.error(item)

            # 即使没有新上传文件，也允许点击按钮重新同步当前知识库。
            try:
                with st.spinner("正在同步本地 Chroma 索引……"):
                    collection = get_cached_collection()
                    sync_report = sync_index(collection=collection)

                st.success(
                    "索引同步完成："
                    f"总片段 {sync_report['total']}，"
                    f"新增 {sync_report['added']}，"
                    f"删除 {sync_report['removed']}，"
                    f"未变化 {sync_report['unchanged']}。"
                )
            except Exception as exc:
                st.error(
                    f"索引同步失败：{type(exc).__name__}。"
                    "请检查 Chroma、Embedding 模型和本地环境。"
                )

        st.divider()
        st.subheader("knowledge/ 当前文件")

        if KNOWLEDGE_DIR.exists():
            current_files = sorted(
                path.name
                for path in KNOWLEDGE_DIR.iterdir()
                if path.is_file() and path.suffix.lower() in {".md", ".txt"}
            )
        else:
            current_files = []

        if current_files:
            for file_name in current_files:
                st.caption("• " + file_name)
        else:
            st.info("当前还没有可读取的 Markdown/TXT 文件。")

    # 主区域：自然语言提问。
    st.subheader("知识库问答")

    with st.form("question_form"):
        question = st.text_input(
            "输入自然语言问题",
            placeholder="例如：工具超时后应该怎样处理？",
        )
        question_submitted = st.form_submit_button(
            "检索并回答",
            type="primary",
            use_container_width=True,
        )

    if question_submitted:
        if not question.strip():
            st.warning("请先输入问题。")
        else:
            with st.spinner("正在检索知识库并检查证据……"):
                result = run_question(question.strip())

            if "error" in result:
                st.error(result["error"])
            else:
                st.session_state["qa_history"].insert(0, result)
                # 页面只保留最近五次结果，避免会话记录无限增长。
                st.session_state["qa_history"] = (
                    st.session_state["qa_history"][:5]
                )

    # 展示最近问答；每条结果都保留本次检索来源与门禁状态。
    if st.session_state["qa_history"]:
        st.divider()
        st.subheader("最近问答记录")

        for item in st.session_state["qa_history"]:
            st.markdown("---")
            st.write("问题：", item["question"])
            st.caption("检索方式：" + item["search_name"])

            if item["fallback_note"]:
                st.info(item["fallback_note"])

            if item["gate_ok"]:
                st.success("证据门禁通过：" + item["gate_reason"])
            else:
                st.warning("证据门禁拒绝：" + item["gate_reason"])

            st.markdown("**本次检索来源**")

            if item["hits"]:
                for index, hit in enumerate(item["hits"], start=1):
                    source = hit.get("source", "未知来源")
                    chunk_id = hit.get("chunk_id", "无 Chunk ID")
                    score = hit.get("score", "无分数")
                    line_start = hit.get("line_start", "?")
                    line_end = hit.get("line_end", "?")

                    with st.expander(
                        f"{index}. {source}｜{chunk_id}"
                    ):
                        st.caption(
                            f"score={score}｜行={line_start}-{line_end}"
                        )
                        st.write(hit.get("text", ""))
            else:
                st.info("本次没有检索到可用片段。")

            st.markdown("**回答**")

            if item["citation_blocked"]:
                st.warning("模型引用校验未通过，回答已被拦截。")

            st.write(item["answer"])

    else:
        st.info(
            "先在左侧上传文档并同步索引，再输入自然语言问题。"
            "如果 Day5 索引已经存在，也可以直接提问。"
        )


if __name__ == "__main__":
    main()