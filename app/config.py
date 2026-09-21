"""全局配置：所有可调参数集中在这里，改配置不用改代码"""
import os
from pathlib import Path

from dotenv import load_dotenv

# 读取项目根目录下的 .env 文件
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# ---- LLM（大模型）配置 ----
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# ---- Embedding（向量化）配置 ----
# bge-small-zh-v1.5：中文效果好、体积小（约 100MB），本地免费运行
# 已下载到项目 models/ 目录，直接用本地路径，不再依赖网络
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", str(BASE_DIR / "models" / "bge-small-zh-v1.5"))
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "pytorch")

# ---- 向量库配置 ----
CHROMA_DIR = str(BASE_DIR / "data" / "vector_store")

# ---- 文档配置 ----
DOCS_DIR = str(BASE_DIR / "data" / "documents")

# ---- 切片参数（面试能讲清楚就赢了） ----
# 256 是 2026-09-18 切分实验的胜出值（14 题命中 12，且在结构化短文档上表现最稳）
# 实验明细见 docs/切分实验结果.md
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "256"))       # 每块约 256 字符
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))  # 相邻块重叠 50 字符，避免切断语义

# ---- 检索参数 ----
TOP_K = 5  # 每次检索返回最相关的 5 个片段
