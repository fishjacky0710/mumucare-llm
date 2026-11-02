FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    git \
    cmake \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# CPU-only torch — 比 CUDA 版本小約 80%，Cloud Run 不需 GPU
RUN pip install --no-cache-dir \
    torch==2.2.0 \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements-api.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements-api.txt

# 在 build 階段預先下載 HuggingFace 模型，避免冷啟動時才下載
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-base')"

COPY . .

# Fail the build if the bundled HNSW snapshot cannot be queried on Linux.
RUN python validate_chroma.py --query

EXPOSE 8080

CMD ["python", "cloud_run_backend.py"]
