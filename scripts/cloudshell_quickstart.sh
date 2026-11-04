#!/usr/bin/env bash
# One command from a fresh clone to a running API, in AWS CloudShell, Azure Cloud Shell or Google Cloud Shell
# (or any Linux/macOS shell). Idempotent: run it again to reuse the venv and the running server.
#
#   ./scripts/cloudshell_quickstart.sh            install, start on 127.0.0.1:8080, smoke test
#   ./scripts/cloudshell_quickstart.sh stop       stop the background server
set -euo pipefail
cd "$(dirname "$0")/.."

PORT="${CONVO_PORT:-8080}"
URL="http://127.0.0.1:${PORT}"
PIDFILE=".convo-server.pid"
LOG=".convo-server.log"

if [[ "${1:-}" == "stop" ]]; then
  [[ -f "$PIDFILE" ]] && kill "$(cat "$PIDFILE")" 2>/dev/null && echo "stopped" || echo "not running"
  rm -f "$PIDFILE"; exit 0
fi

# ---- which cloud shell? ------------------------------------------------------------------------------
if [[ "${AWS_EXECUTION_ENV:-}" == "CloudShell" ]]; then
  CLOUD=aws; EXTRAS=bedrock
elif [[ -n "${ACC_CLOUD:-}" || "${AZUREPS_HOST_ENVIRONMENT:-}" == cloud-shell* ]]; then
  CLOUD=azure; EXTRAS=azure
elif [[ "${CLOUD_SHELL:-}" == "true" || -n "${DEVSHELL_PROJECT_ID:-}" ]]; then
  CLOUD=gcp; EXTRAS=gemini
else
  CLOUD=local; EXTRAS=all
fi
EXTRAS="${CONVO_EXTRAS:-$EXTRAS}"
echo "▸ environment: $CLOUD  (installing extras: $EXTRAS)"

# ---- Python 3.10+ -----------------------------------------------------------------------------------
PY=""
for p in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$p" >/dev/null && "$p" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then
    PY="$(command -v "$p")"; break
  fi
done

if [[ ! -x .venv/bin/python ]]; then
  if [[ -n "$PY" ]]; then
    echo "▸ creating .venv with $("$PY" --version)"
    "$PY" -m venv .venv
  elif [[ "$CLOUD" == "aws" ]] && command -v dnf >/dev/null; then
    echo "▸ installing python3.11 (AWS CloudShell ships 3.9)"
    sudo dnf install -y -q python3.11 >/dev/null
    python3.11 -m venv .venv
  else
    echo "▸ no Python 3.10+ found; fetching one with uv"
    command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
    export PATH="$HOME/.local/bin:$PATH"
    uv venv -q --seed --python 3.12 .venv
  fi
fi

echo "▸ installing package"
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -e ".[$EXTRAS]"

# ---- start (or reuse) the server ---------------------------------------------------------------------
if curl -sf "$URL/healthz" >/dev/null 2>&1; then
  echo "▸ server already running on $URL"
else
  echo "▸ starting server on $URL (logs: $LOG)"
  CONVO_PORT="$PORT" nohup .venv/bin/convo serve >"$LOG" 2>&1 &
  echo $! >"$PIDFILE"
  for _ in $(seq 1 30); do curl -sf "$URL/healthz" >/dev/null 2>&1 && break; sleep 0.5; done
  curl -sf "$URL/healthz" >/dev/null || { echo "server did not start; see $LOG"; tail -20 "$LOG"; exit 1; }
fi

# ---- smoke test --------------------------------------------------------------------------------------
export CONVO_URL="$URL"
echo; .venv/bin/convo models
echo; .venv/bin/convo ask "Hello from $CLOUD" -m offline --no-stream

# ---- next steps --------------------------------------------------------------------------------------
cat <<EOF

✔ API running at $URL   (OpenAPI docs: $URL/docs)

Add the CLI to this shell:   source .venv/bin/activate && export CONVO_URL=$URL
Interactive chat:            convo chat -m <model>      (inside: /model <name> swaps, history kept)
No-install client:           ./scripts/convo.sh ask "hi" <model>
Stop the server:             ./scripts/cloudshell_quickstart.sh stop
EOF

case "$CLOUD" in
  aws) cat <<'EOF'

AWS CloudShell: Bedrock credentials are already here. Try:
  convo ask "Explain the CAP theorem in 3 lines" -m smart
  (needs Bedrock model access for Claude in your region; AWS_REGION picks the region)
EOF
  ;;
  azure) cat <<'EOF'

Azure Cloud Shell: point at your Azure OpenAI resource (your az login is used; no key needed if you
have the "Cognitive Services OpenAI User" role), then restart:
  export AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com AZURE_OPENAI_DEPLOYMENT=<deployment>
  ./scripts/cloudshell_quickstart.sh stop && ./scripts/cloudshell_quickstart.sh
  convo ask "hi" -m azure-gpt
EOF
  ;;
  gcp) cat <<'EOF'

Google Cloud Shell: Vertex AI works with your login (enable it once: gcloud services enable aiplatform.googleapis.com).
  convo ask "hi" -m gemini-vertex
  Web Preview → port 8080 → /docs opens the interactive API docs.
EOF
  ;;
esac
