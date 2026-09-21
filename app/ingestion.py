"""数据接入层：文档 → 解析 → 切片（闭环的第①②步）"""
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_documents(docs_dir: str):
    """读取 docs_dir 下所有 .md / .txt 文件，返回 Document 列表。

    每个 Document 包含 page_content（文本）和 metadata（如 source 来源路径）。
    """
    docs_path = Path(docs_dir)
    file_paths = sorted(
        path for path in docs_path.rglob("*")
        if path.is_file() and path.suffix.lower() in {".md", ".txt"}
    )

    docs = []
    for path in file_paths:
        # TextLoader 会把实际文件路径写入 metadata["source"]，供后续引用溯源。
        docs.extend(TextLoader(str(path), encoding="utf-8").load())

    if not docs:
        raise RuntimeError(
            f"未在 {docs_dir} 下找到任何 .md/.txt 文档，请先放入资料"
        )
    return docs


def split_documents(docs, chunk_size: int = 400, chunk_overlap: int = 50):
    """把长文档切成小块。关键参数：
    - chunk_size: 每块多长（字符数）
    - chunk_overlap: 相邻块重叠多少，防止关键信息被拦腰切断
    - separators: 优先在段落/句子边界处切，中文加了句号分号等
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )
    return splitter.split_documents(docs)
