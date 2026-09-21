"""一次性脚本：从镜像站完整下载 BGE embedding 模型到项目本地。

用法：HF_ENDPOINT=https://hf-mirror.com python download_model.py
下载完成后，config.py 里的 EMBEDDING_MODEL 指向本地路径即可。
"""
import os
from pathlib import Path

from huggingface_hub import snapshot_download

BASE_DIR = Path(__file__).resolve().parent
LOCAL_DIR = BASE_DIR / "models" / "bge-small-zh-v1.5"

print("📥 开始下载 BAAI/bge-small-zh-v1.5 到项目 models/ 目录...")
print(f"   目标: {LOCAL_DIR}")

LOCAL_DIR.mkdir(parents=True, exist_ok=True)
path = snapshot_download(
    repo_id="BAAI/bge-small-zh-v1.5",
    local_dir=str(LOCAL_DIR),
)
print(f"✅ 下载完成: {path}")
