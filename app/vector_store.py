"""向量化 + 向量存储（闭环的第③④步）"""
from typing import List

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


class BgeChineseEmbeddings(HuggingFaceEmbeddings):
    """保留 BGE 中文模型推荐的「查询指令、文档无指令」编码方式。

    ``HuggingFaceBgeEmbeddings`` 已弃用，但通用的
    ``HuggingFaceEmbeddings`` 不会自动添加 BGE 的查询指令。这里显式保留
    原有行为，避免升级依赖后悄悄改变检索质量和历史实验基线。
    """

    query_instruction: str = "为这个句子生成表示以用于检索相关文章："

    def embed_query(self, text: str) -> List[float]:
        return super().embed_query(f"{self.query_instruction}{text}")


def get_embeddings(model_name: str = "BAAI/bge-small-zh-v1.5"):
    """加载本地 embedding 模型，把文本变成向量。"""
    return BgeChineseEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},          # 没有 GPU 也能跑
        encode_kwargs={"normalize_embeddings": True},  # 归一化，余弦相似度更快
    )


def get_vector_store(embeddings, persist_dir: str) -> Chroma:
    """连接（或创建）本地 Chroma 向量库。索引存在磁盘上，重启不丢。"""
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)


def get_index_chunk_count(persist_dir: str) -> int:
    """仅读取本地索引的切片数，不加载模型，也不调用任何 LLM API。"""
    vector_store = Chroma(persist_directory=persist_dir)
    return vector_store._collection.count()


def build_index(docs, embeddings, persist_dir: str) -> Chroma:
    """把切片文档向量化并写入向量库（首次建索引用）。"""
    return Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
