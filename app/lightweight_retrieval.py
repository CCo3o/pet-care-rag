"""免费云实例的轻量检索：不加载向量模型，按中文字符片段匹配业务文档。"""
from pathlib import Path

from langchain_core.documents import Document


def _terms(text: str) -> set[str]:
    chars = [char for char in text if '\u4e00' <= char <= '\u9fff']
    singles = {char for char in chars if char not in '的了是吗我你他她它和与及在有'}
    pairs = {''.join(chars[index:index + 2]) for index in range(len(chars) - 1)}
    return singles | pairs


class LightweightRetriever:
    def __init__(self, docs_dir: str, chunk_size: int = 256, overlap: int = 50, k: int = 5):
        self.k = k
        self.docs = []
        for path in sorted(Path(docs_dir).rglob('*')):
            if not path.is_file() or path.suffix.lower() not in {'.md', '.txt'}:
                continue
            text = path.read_text(encoding='utf-8')
            step = max(1, chunk_size - overlap)
            for start in range(0, len(text), step):
                content = text[start:start + chunk_size]
                if content.strip():
                    self.docs.append(Document(page_content=content, metadata={'source': str(path)}))

    def invoke(self, question: str):
        terms = _terms(question)
        scored = []
        for index, doc in enumerate(self.docs):
            content = doc.page_content
            score = sum(len(term) ** 2 * content.count(term) for term in terms)
            if score:
                scored.append((score, -index, doc))
        return [item[2] for item in sorted(scored, reverse=True)[:self.k]]
