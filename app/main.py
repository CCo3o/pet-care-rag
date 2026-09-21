"""宠物寄养 RAG 助手 · 基础版命令行入口

运行方式（在项目根目录 pet-care-rag 下执行）：
    python -m app.main              # 首次：自动建索引 + 进入问答
    python -m app.main --rebuild    # 改了文档后，强制重建索引

步骤演示：
    ❓ 你的问题: 猫可以吃巧克力吗？
    💬 （基于资料的回答）
    📎 参考来源：
      - 《猫可以吃巧克力吗》
"""
import argparse
import os
import sqlite3

from openai import APIConnectionError, APIStatusError, APITimeoutError

from app.chat import ChatAssistant
from app.display import render_reply

from app.config import (
    CHROMA_DIR, DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP,
    EMBEDDING_MODEL, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
    TOP_K,
)
from app.ingestion import load_documents, split_documents
from app.vector_store import (
    build_index,
    get_embeddings,
    get_index_chunk_count,
    get_vector_store,
)
from app.retrieval import (
    create_llm, create_retriever,
    PROMPT_TEMPLATE,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def build_index_from_documents(embeddings):
    """读取资料、切片并创建一个新的向量索引。"""
    print("📥 正在读取文档并建立索引（可能需要几分钟）...")
    docs = load_documents(DOCS_DIR)
    chunks = split_documents(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    vector_store = build_index(chunks, embeddings, CHROMA_DIR)
    print(f"✅ 索引完成：{len(chunks)} 个切片已入库\n")
    return vector_store


def rebuild_index(embeddings):
    """删除整个旧索引后再创建，避免 Chroma 把新切片追加到旧集合。"""
    import shutil

    if os.path.exists(CHROMA_DIR):
        print("🗑️ 强制重建：正在清空旧索引...")
        shutil.rmtree(CHROMA_DIR)
    return build_index_from_documents(embeddings)


def ensure_index(embeddings):
    """复用已有索引；不存在或为空时才创建。"""
    vector_store = get_vector_store(embeddings, CHROMA_DIR)
    if vector_store._collection.count() > 0:
        return vector_store
    return build_index_from_documents(embeddings)


def show_index_status():
    """展示知识库与索引状态，用于确认重建是否真的生效。"""
    document_count = len(load_documents(DOCS_DIR))
    chunk_count = get_index_chunk_count(CHROMA_DIR) if os.path.exists(CHROMA_DIR) else 0

    print("📊 知识库状态")
    print(f"   - 资料文件：{document_count} 篇")
    print(f"   - 当前索引：{chunk_count} 个切片")
    if chunk_count:
        print("   - 提示：连续执行 --rebuild 后，切片数应保持不变，而不是持续增加。")
    else:
        print("   - 提示：尚未建立索引，请运行 python -m app.main --rebuild")


def main():
    parser = argparse.ArgumentParser(description="宠物寄养 RAG 助手")
    parser.add_argument("--rebuild", action="store_true", help="强制重建索引")
    parser.add_argument("--status", action="store_true", help="查看资料与索引切片数量（不调用 API）")
    parser.add_argument("--debug", action="store_true", help="显示工具选择、参数和执行结果，便于学习和排错")
    args = parser.parse_args()

    # --status 只读本地文件，不需要模型或 DeepSeek API Key。
    if args.status:
        show_index_status()
        return

    # 1. 加载 embedding 模型；--rebuild 必须在连接旧库前删除它，
    #    否则 Chroma.from_documents 会向旧集合追加数据。
    embeddings = get_embeddings(EMBEDDING_MODEL)

    # 2. 建索引（首次 或 --rebuild）
    if args.rebuild:
        print("🔄 正在重建索引...")
        vector_store = rebuild_index(embeddings)
    else:
        vector_store = ensure_index(embeddings)

    # 3. 只有进入问答时才需要 DeepSeek Key；索引重建完全在本地完成。
    if not DEEPSEEK_API_KEY:
        print("❌ 索引已就绪，但没有找到 DEEPSEEK_API_KEY，无法进入问答模式")
        print("   请复制 .env.example 为 .env，填入你在 platform.deepseek.com 申请的 Key")
        return

    # 4. 组装问答链路：问题 → 检索 → 拼 prompt → LLM 回答
    llm = create_llm(DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL)
    retriever = create_retriever(vector_store, k=TOP_K)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    answer_chain = prompt | llm | StrOutputParser()
    assistant = ChatAssistant(llm, retriever, answer_chain)

    # 5. 进入交互问答
    print("🐱🐶 宠物寄养助手已就绪！可询价、查资料、查余位及保存本地演示预订。")
    print("输入 /clear 只清空聊天（不取消已保存预订），exit 退出。预订结束日也占位，次日释放。")
    if args.debug:
        print("🔎 调试模式：显示模型的工具请求和程序执行结果；每次提问会调用 DeepSeek API。")
    while True:
        try:
            question = input("\n❓ 你的问题: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见！")
            break
        if question.lower() in ("exit", "quit", "q"):
            print("👋 再见！")
            break
        if not question:
            continue
        if question == "/clear":
            assistant.clear()
            print("🧹 当前聊天已清空，已保存的预订仍然保留。")
            continue
        try:
            reply = assistant.reply(question)
        except (APITimeoutError, APIConnectionError):
            print("💬 暂时无法连接模型服务，本次未完成。请稍后重试，或输入 exit 退出。")
            continue
        except APIStatusError as exc:
            print(f"💬 模型服务返回错误（HTTP {exc.status_code}），请检查配置或额度后重试。")
            continue
        except sqlite3.Error:
            print("💬 预订数据库暂时无法完成操作，请稍后重试；当前没有确认新的预订成功。")
            continue

        print(render_reply(reply, debug=args.debug))


if __name__ == "__main__":
    main()
