FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.cache/huggingface \
    RETRIEVAL_BACKEND=lightweight

WORKDIR /app
COPY requirements-cloud.txt .
RUN pip install --no-cache-dir -r requirements-cloud.txt
COPY . .

EXPOSE 8000
CMD ["python", "-m", "app.web"]
