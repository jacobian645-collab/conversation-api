#!/usr/bin/env bash
# Deploy to Cloud Run from Google Cloud Shell, private by default, Gemini on Vertex AI as the default model.
#
#   ./deploy/gcp_cloud_run.sh
#
# Two layers of auth on the deployed service:
#   1. Cloud Run IAM (--no-allow-unauthenticated): callers need roles/run.invoker and an identity token
#   2. the app's API key, stored in Secret Manager and injected as CONVO_API_KEY
set -euo pipefail
cd "$(dirname "$0")/.."

PROJECT="${GOOGLE_CLOUD_PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-conversation-api}"
SECRET="${SECRET:-convo-api-key}"
DEFAULT_MODEL="${CONVO_DEFAULT_MODEL:-gemini-vertex}"
: "${PROJECT:?set GOOGLE_CLOUD_PROJECT or run: gcloud config set project <id>}"

echo "▸ project $PROJECT · region $REGION · service $SERVICE"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com \
  secretmanager.googleapis.com aiplatform.googleapis.com --project "$PROJECT"

# Runtime identity: a dedicated service account with only what the app needs.
SA="${SERVICE}-sa@${PROJECT}.iam.gserviceaccount.com"
gcloud iam service-accounts describe "$SA" --project "$PROJECT" >/dev/null 2>&1 \
  || gcloud iam service-accounts create "${SERVICE}-sa" --project "$PROJECT" --display-name "$SERVICE runtime"
gcloud projects add-iam-policy-binding "$PROJECT" --member "serviceAccount:$SA" \
  --role roles/aiplatform.user --condition None >/dev/null

# API key in Secret Manager (created once, reused on redeploys).
if ! gcloud secrets describe "$SECRET" --project "$PROJECT" >/dev/null 2>&1; then
  openssl rand -hex 24 | tr -d '\n' | gcloud secrets create "$SECRET" --project "$PROJECT" --data-file=-
fi
gcloud secrets add-iam-policy-binding "$SECRET" --project "$PROJECT" --member "serviceAccount:$SA" \
  --role roles/secretmanager.secretAccessor >/dev/null

gcloud run deploy "$SERVICE" --source . --project "$PROJECT" --region "$REGION" \
  --service-account "$SA" --no-allow-unauthenticated \
  --cpu 1 --memory 512Mi --min-instances 0 --max-instances 3 --timeout 300 \
  --set-secrets "CONVO_API_KEY=${SECRET}:latest" \
  --set-env-vars "CONVO_DEFAULT_MODEL=${DEFAULT_MODEL},GOOGLE_CLOUD_PROJECT=${PROJECT},GOOGLE_CLOUD_LOCATION=${REGION}"

URL="$(gcloud run services describe "$SERVICE" --project "$PROJECT" --region "$REGION" --format 'value(status.url)')"
cat <<EOF

✔ deployed: $URL

Call it from Cloud Shell:
  export CONVO_URL=$URL
  export CONVO_API_KEY=\$(gcloud secrets versions access latest --secret $SECRET --project $PROJECT)
  export CONVO_BEARER_TOKEN=\$(gcloud auth print-identity-token)
  convo ask "hello from Cloud Run"          # or ./scripts/convo.sh ask "hello"

Conversations are in memory per instance. For durable, multi-instance history, back the store with
Firestore or Memorystore (see docs/adr/0002-conversation-state.md).
EOF
