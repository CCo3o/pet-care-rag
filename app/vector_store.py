"""向量化 + 向量存储（闭环的第③④步）"""
import os
from typing import List

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings


class BgeChineseEmbeddings:
    """保留 BGE 中文模型推荐的「查询指令、文档无指令」编码方式。

    ``HuggingFaceBgeEmbeddings`` 已弃用，但通用的
    ``HuggingFaceEmbeddings`` 不会自动添加 BGE 的查询指令。这里显式保留
    原有行为，避免升级依赖后悄悄改变检索质量和历史实验基线。
    """

    query_instruction: str = "为这个句子生成表示以用于检索相关文章："

    def __init__(self, model_name: str):
        # 延迟导入：云端的 ONNX 部署不安装 PyTorch / sentence-transformers。
        from langchain_huggingface import HuggingFaceEmbeddings
        self._embedding = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embedding.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embedding.embed_query(f"{self.query_instruction}{text}")


class FastEmbedChineseEmbeddings(Embeddings):
    """云端轻量向量化：ONNX Runtime 替代 PyTorch，降低容器内存。"""

    def __init__(self, model_name: str):
        from fastembed import TextEmbedding
        self.model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [vector.tolist() for vector in self.model.passage_embed(texts)]

    def embed_query(self, text: str) -> List[float]:
        return next(self.model.query_embed(text)).tolist()


def get_embeddings(model_name: str = "BAAI/bge-small-zh-v1.5"):
    """本地保留 PyTorch 基线；云端可用 EMBEDDING_BACKEND=fastembed 切换 ONNX。"""
    if os.getenv("EMBEDDING_BACKEND", "pytorch").lower() == "fastembed":
        return FastEmbedChineseEmbeddings(model_name)
    return BgeChineseEmbeddings(model_name)


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
