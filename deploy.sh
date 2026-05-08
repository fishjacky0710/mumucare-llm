#!/usr/bin/env bash
# ============================================================
# Cloud Run 部署腳本
# 費用最精簡策略：
#   - min-instances=0  → 無人使用時縮到 0，不計費
#   - max-instances=2  → 少數人同時使用時可開第 2 台，閒置仍縮到 0
#   - CPU 只在處理 request 時計費（Cloud Run 預設行為）
# ============================================================
set -euo pipefail

# -------- 必填：請填入你的 GCP 專案 ID --------
PROJECT_ID="${GCP_PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
if [ -z "$PROJECT_ID" ]; then
  echo "❌ 請設定 GCP_PROJECT_ID 環境變數，或先執行 gcloud config set project <PROJECT_ID>"
  exit 1
fi

# -------- 設定 --------
SERVICE_NAME="mumucare-llm"
REGION="asia-east1"
IMAGE="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

# OPENAI_API_KEY 必須已設定
if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "❌ 請先設定 OPENAI_API_KEY 環境變數"
  exit 1
fi

echo "========================================"
echo "  PROJECT : $PROJECT_ID"
echo "  SERVICE : $SERVICE_NAME"
echo "  REGION  : $REGION"
echo "  IMAGE   : $IMAGE"
echo "========================================"

# -------- 1. Build & push --------
echo ""
echo "▶ 建置 Docker image 並推送至 GCR..."
gcloud builds submit \
  --tag "$IMAGE" \
  --project "$PROJECT_ID"

# -------- 2. Deploy --------
echo ""
echo "▶ 部署至 Cloud Run..."
gcloud run deploy "$SERVICE_NAME" \
  --image "$IMAGE" \
  --region "$REGION" \
  --project "$PROJECT_ID" \
  --platform managed \
  --allow-unauthenticated \
  --min-instances 0 \
  --max-instances 2 \
  --memory 4Gi \
  --cpu 2 \
  --timeout 300 \
  --set-env-vars "OPENAI_API_KEY=${OPENAI_API_KEY}"

echo ""
echo "✅ 部署完成！"
gcloud run services describe "$SERVICE_NAME" \
  --region "$REGION" \
  --project "$PROJECT_ID" \
  --format "value(status.url)"
