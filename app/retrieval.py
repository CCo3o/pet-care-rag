"""检索 + 生成（闭环的第⑤⑥⑦步）"""
from langchain_openai import ChatOpenAI


# Prompt 模板：核心是"只根据资料回答，没有就直说"，从源头抑制幻觉
PROMPT_TEMPLATE = """你是一个温馨的宠物寄养店的专业客服，回答要亲切、耐心。负责回答客户关于宠物寄养、喂养、护理的问题。

请严格基于下面的参考资料回答问题：
1. 参考资料里有的内容，要准确、具体地回答；
2. 参考资料里没有的内容，直接说"资料中没有相关内容"，绝对不要编造；
3. 涉及用药、就医的内容，提醒"如有异常请及时联系宠物医院"。
4. 本链路只解释知识和规则；具体寄养总价需要算价工具，不要在这里自行计算。

参考资料：
{context}

用户问题：{question}
"""


def create_llm(api_key: str, base_url: str, model: str) -> ChatOpenAI:
    """创建大模型客户端。DeepSeek 提供 OpenAI 兼容接口，所以用 ChatOpenAI 就能接。"""
    return ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=0.3,  # 偏低，让回答更稳定、更忠于资料
        timeout=30,
        max_retries=0,  # 出错返回给用户重试，避免命令行长时间无响应
    )


def create_retriever(vector_store, k: int = 5):
    """创建检索器：输入问题 → 输出向量库里最相关的 k 个片段。"""
    return vector_store.as_retriever(search_kwargs={"k": k})


def format_docs(docs) -> str:
    """把检索到的片段拼成带来源的上下文，供 LLM 阅读。"""
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "未知来源")
        parts.append(f"[片段{i} | 来源: {source}]\n{doc.page_content}")
    return "\n\n".join(parts)


def format_sources(docs) -> list:
    """提取每个片段的来源路径，去重后展示给用户（引用溯源）。"""
    seen, sources = set(), []
    for doc in docs:
        src = doc.metadata.get("source", "未知来源")
        if src not in seen:
            seen.add(src)
            sources.append(src)
    return sources
