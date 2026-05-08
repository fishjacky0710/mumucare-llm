FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential gcc git \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch — 比 CUDA 版本小約 80%，Cloud Run 不需 GPU
RUN pip install --no-cache-dir \
    torch==2.2.0 \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 在 build 階段預先下載 HuggingFace 模型，避免冷啟動時才下載
ENV HF_HOME=/app/.cache/huggingface
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-base')"
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('jinaai/jina-reranker-v2-base-multilingual', trust_remote_code=True)"

COPY . .

ENV PORT=8080
EXPOSE 8080

CMD ["python", "main.py"]
