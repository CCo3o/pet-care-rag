FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app
COPY requirements.txt .
# 云端只使用 CPU 做向量化；先安装 CPU 版 PyTorch，避免 sentence-transformers 拉取数 GB 的 CUDA 依赖。
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch && \
    pip install --no-cache-dir -r requirements.txt
COPY . .

EXPOSE 8000
CMD ["python", "-m", "app.web"]
