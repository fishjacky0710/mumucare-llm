#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

PROJECT_ID="${GCP_PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${GCP_REGION:-asia-east1}"
SERVICE_NAME="${CLOUD_RUN_SERVICE_NAME:-mumucare-llm}"
ARTIFACT_REPOSITORY="${ARTIFACT_REPOSITORY:-mumucare-llm}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REPOSITORY}/${SERVICE_NAME}"
FIRESTORE_PROJECT_ID="${FIRESTORE_PROJECT_ID:-mumu-care}"
RUNTIME_SERVICE_ACCOUNT_NAME="${RUNTIME_SERVICE_ACCOUNT_NAME:-mumucare-llm-runtime}"
RUNTIME_SERVICE_ACCOUNT_EMAIL="${RUNTIME_SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
OPENAI_SECRET_NAME="${OPENAI_SECRET_NAME:-mumucare-openai-api-key}"
FIREBASE_AUTH_PROJECT_ID="${FIREBASE_AUTH_PROJECT_ID:-mumu-care}"
EMPLOYEE_EMAIL_DOMAIN="${EMPLOYEE_EMAIL_DOMAIN:-maria.org.tw}"
CORS_ALLOW_ORIGINS="${CORS_ALLOW_ORIGINS:-https://mumu-care.web.app,https://mumu-care.firebaseapp.com,https://mumu-care--jacky-lryp8mk8.web.app}"
export RERANKER_ENABLED="${RERANKER_ENABLED:-false}"

if [[ -z "$PROJECT_ID" || "$PROJECT_ID" == "(unset)" ]]; then
  echo "錯誤：請設定 GCP_PROJECT_ID，或先執行 gcloud config set project PROJECT_ID。" >&2
  exit 1
fi

if [[ -n "${CORS_ALLOW_ORIGINS:-}" ]]; then
  IFS=',' read -ra ORIGINS <<< "$CORS_ALLOW_ORIGINS"
  for origin in "${ORIGINS[@]}"; do
    trimmed="${origin#"${origin%%[![:space:]]*}"}"
    trimmed="${trimmed%"${trimmed##*[![:space:]]}"}"
    if [[ "$trimmed" != https://* || "$trimmed" == "*" ]]; then
      echo "錯誤：正式環境的 CORS_ALLOW_ORIGINS 只能包含明確的 HTTPS 網址。" >&2
      exit 1
    fi
  done
fi

if ! gcloud iam service-accounts describe "$RUNTIME_SERVICE_ACCOUNT_EMAIL" \
  --project "$PROJECT_ID" >/dev/null 2>&1; then
  echo "錯誤：找不到執行服務帳號。請先執行 ./setup_cloud_security.sh。" >&2
  exit 1
fi

for secret_name in "$OPENAI_SECRET_NAME"; do
  if ! gcloud secrets describe "$secret_name" --project "$PROJECT_ID" >/dev/null 2>&1; then
    echo "錯誤：找不到 Secret Manager secret：${secret_name}。請先執行 ./setup_cloud_security.sh。" >&2
    exit 1
  fi
done

python3 validate_chroma.py

echo "Building ${IMAGE} in ${REGION} ..."
gcloud builds submit --region "$REGION" \
  --default-buckets-behavior regional-user-owned-bucket \
  --config cloudbuild-api.yaml \
  --substitutions "_IMAGE=${IMAGE}" --project "$PROJECT_ID"

echo "Deploying ${SERVICE_NAME} to ${REGION} ..."
# Browser requests are authenticated and authorized using Firebase employee ID tokens.
gcloud run deploy "$SERVICE_NAME" \
  --image "$IMAGE" \
  --region "$REGION" \
  --platform managed \
  --allow-unauthenticated \
  --service-account "$RUNTIME_SERVICE_ACCOUNT_EMAIL" \
  --execution-environment gen2 \
  --cpu 2 \
  --memory 4Gi \
  --cpu-throttling \
  --min-instances 0 \
  --max-instances 5 \
  --concurrency 1 \
  --timeout 300 \
  --set-secrets "OPENAI_API_KEY=${OPENAI_SECRET_NAME}:latest" \
  --set-env-vars "^@^APP_ENV=production@FIRESTORE_PROJECT_ID=${FIRESTORE_PROJECT_ID}@FIREBASE_AUTH_PROJECT_ID=${FIREBASE_AUTH_PROJECT_ID}@EMPLOYEE_EMAIL_DOMAIN=${EMPLOYEE_EMAIL_DOMAIN}@CORS_ALLOW_ORIGINS=${CORS_ALLOW_ORIGINS}@MAX_REQUEST_BYTES=262144@RATE_LIMIT_PER_MINUTE=30@RERANKER_ENABLED=${RERANKER_ENABLED}" \
  --project "$PROJECT_ID"

echo "部署完成。前端請使用 Authorization: Bearer <Firebase ID Token>。"
