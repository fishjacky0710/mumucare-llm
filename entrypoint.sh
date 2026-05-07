#!/usr/bin/env bash
set -euo pipefail

echo "==================== [boot] entrypoint start ===================="

PORT="${PORT:-8080}"
LLAMA_CTX="${LLAMA_CTX:-1024}"
LLAMA_THREADS="${LLAMA_THREADS:-4}"

# 你現在 bucket 掛載到 /mnt/models，模型實際在 /mnt/models/student-merged-q5_k_m.gguf
: "${MODEL_PATH:=/mnt/models/student-merged-q5_k_m.gguf}"

echo "[boot] PORT=$PORT"
echo "[boot] LLAMA_CTX=$LLAMA_CTX"
echo "[boot] LLAMA_THREADS=$LLAMA_THREADS"
echo "[boot] MODEL_PATH=$MODEL_PATH"

echo "[boot] uname -a:"
uname -a || true

echo "[boot] listing /app:"
ls -lah /app || true

if [ ! -x /app/llama-server ]; then
  echo "[boot] ERROR: /app/llama-server not executable" >&2
  exit 1
fi

echo "[boot] listing /mnt:"
ls -lah /mnt || true

if [ ! -d /mnt/models ]; then
  echo "[boot] ERROR: /mnt/models not found (GCS volume not mounted?)" >&2
  exit 1
fi

echo "[boot] listing /mnt/models:"
ls -lah /mnt/models || true

if [ ! -f "$MODEL_PATH" ]; then
  echo "[boot] ERROR: model file does not exist: $MODEL_PATH" >&2
  echo "[boot] hint: your file is under /mnt/models/ (NOT /mnt/models/models/)" >&2
  exit 1
fi

echo "[boot] model file size:"
ls -lh "$MODEL_PATH" || true

echo "==================== [boot] starting llama-server ===================="

exec /app/llama-server \
  -m "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --ctx-size "$LLAMA_CTX" \
  --threads "$LLAMA_THREADS"