#!/usr/bin/env bash
set -euo pipefail

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set. Put it in .env or export it before running." >&2
  exit 1
fi

if [[ -z "${GRADIO_PASSWORD:-}" ]]; then
  echo "GRADIO_PASSWORD is not set. Put it in .env or export it before running." >&2
  exit 1
fi

gcloud run deploy mumucare-gradio \
  --source . \
  --project mumucare-llm \
  --region asia-east1 \
  --memory 4Gi \
  --cpu 1 \
  --min-instances 0 \
  --max-instances 1 \
  --concurrency 1 \
  --timeout 600 \
  --allow-unauthenticated \
  --set-env-vars OPENAI_API_KEY="${OPENAI_API_KEY}",GRADIO_USERNAME="${GRADIO_USERNAME:-maria}",GRADIO_PASSWORD="${GRADIO_PASSWORD}"
