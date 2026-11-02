#!/usr/bin/env bash
set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

PROJECT_ID="${GCP_PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
RUNTIME_SERVICE_ACCOUNT_NAME="${RUNTIME_SERVICE_ACCOUNT_NAME:-mumucare-llm-runtime}"
RUNTIME_SERVICE_ACCOUNT_EMAIL="${RUNTIME_SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
OPENAI_SECRET_NAME="${OPENAI_SECRET_NAME:-mumucare-openai-api-key}"
FIRESTORE_PROJECT_ID="${FIRESTORE_PROJECT_ID:-mumu-care}"
FIREBASE_AUTH_PROJECT_ID="${FIREBASE_AUTH_PROJECT_ID:-mumu-care}"
REGION="${GCP_REGION:-asia-east1}"
ARTIFACT_REPOSITORY="${ARTIFACT_REPOSITORY:-mumucare-llm}"

if [[ -z "$PROJECT_ID" || "$PROJECT_ID" == "(unset)" ]]; then
  echo "錯誤：請設定 GCP_PROJECT_ID，或先執行 gcloud config set project PROJECT_ID。" >&2
  exit 1
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "錯誤：請先在 .env 設定 OPENAI_API_KEY。" >&2
  exit 1
fi

echo "Enabling required Google Cloud APIs ..."
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  secretmanager.googleapis.com \
  firestore.googleapis.com \
  identitytoolkit.googleapis.com \
  artifactregistry.googleapis.com \
  --project "$PROJECT_ID"

if ! gcloud iam service-accounts describe "$RUNTIME_SERVICE_ACCOUNT_EMAIL" \
  --project "$PROJECT_ID" >/dev/null 2>&1; then
  gcloud iam service-accounts create "$RUNTIME_SERVICE_ACCOUNT_NAME" \
    --display-name "Mumucare LLM Cloud Run runtime" \
    --project "$PROJECT_ID"
fi

# 後端目前只讀取 Firestore，因此不授予寫入權限。
if [[ "${CONFIGURE_FIRESTORE_IAM:-true}" == "true" ]]; then
  gcloud projects add-iam-policy-binding "$FIRESTORE_PROJECT_ID" \
    --member "serviceAccount:${RUNTIME_SERVICE_ACCOUNT_EMAIL}" \
    --role "roles/datastore.viewer" \
    --condition=None >/dev/null
  gcloud projects add-iam-policy-binding "$FIREBASE_AUTH_PROJECT_ID" \
    --member "serviceAccount:${RUNTIME_SERVICE_ACCOUNT_EMAIL}" \
    --role "roles/firebaseauth.viewer" \
    --condition=None >/dev/null
else
  echo "已略過 IAM 設定，請由資料專案管理員授予 Firestore 與 Firebase Authentication 唯讀權限。"
fi

if ! gcloud artifacts repositories describe "$ARTIFACT_REPOSITORY" \
  --location "$REGION" --project "$PROJECT_ID" >/dev/null 2>&1; then
  gcloud artifacts repositories create "$ARTIFACT_REPOSITORY" \
    --repository-format=docker --location "$REGION" --project "$PROJECT_ID"
fi

upsert_secret() {
  local secret_name="$1"
  local secret_value="$2"

  if gcloud secrets describe "$secret_name" --project "$PROJECT_ID" >/dev/null 2>&1; then
    printf '%s' "$secret_value" | gcloud secrets versions add "$secret_name" \
      --data-file=- \
      --project "$PROJECT_ID" >/dev/null
  else
    printf '%s' "$secret_value" | gcloud secrets create "$secret_name" \
      --replication-policy=automatic \
      --data-file=- \
      --project "$PROJECT_ID" >/dev/null
  fi

  gcloud secrets add-iam-policy-binding "$secret_name" \
    --member "serviceAccount:${RUNTIME_SERVICE_ACCOUNT_EMAIL}" \
    --role "roles/secretmanager.secretAccessor" \
    --project "$PROJECT_ID" >/dev/null
}

upsert_secret "$OPENAI_SECRET_NAME" "$OPENAI_API_KEY"

echo "安全初始化完成："
echo "- Cloud Run 使用專用服務帳號：${RUNTIME_SERVICE_ACCOUNT_EMAIL}"
echo "- Firestore 專案：${FIRESTORE_PROJECT_ID}（IAM 設定：${CONFIGURE_FIRESTORE_IAM:-true}）"
echo "- OpenAI 金鑰：已存入 Secret Manager（未輸出金鑰內容）"
echo "- 前端驗證：Firebase ID Token（不再使用共用 API key）"
echo "接著執行 ./deploy.sh。"
