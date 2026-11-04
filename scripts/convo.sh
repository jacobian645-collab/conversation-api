#!/usr/bin/env bash
# Zero-install client: curl + python3 (present in AWS CloudShell, Azure Cloud Shell and Google Cloud Shell).
#
#   ./scripts/convo.sh models
#   ./scripts/convo.sh ask "What is RAG?" [model]          start a conversation (prints its id)
#   ./scripts/convo.sh say <id> "Shorter please" [model]   continue; a model here applies to this turn only
#   ./scripts/convo.sh swap <id> <model>                   change the model for later turns (history kept)
#   ./scripts/convo.sh stream "Explain SSE" [model]        raw Server-Sent Events
#   ./scripts/convo.sh history <id>
#
# Env: CONVO_URL (default http://127.0.0.1:8080), CONVO_API_KEY, CONVO_BEARER_TOKEN
set -euo pipefail

URL="${CONVO_URL:-http://127.0.0.1:8080}"
HDR=(-H "Content-Type: application/json")
[[ -n "${CONVO_API_KEY:-}" ]] && HDR+=(-H "X-API-Key: ${CONVO_API_KEY}")
[[ -n "${CONVO_BEARER_TOKEN:-}" ]] && HDR+=(-H "Authorization: Bearer ${CONVO_BEARER_TOKEN}")

# Build a JSON body from key=value pairs, skipping empty values. Values are JSON-escaped by python.
json() { python3 -c 'import json,sys; print(json.dumps({k: v for k, v in (a.split("=", 1) for a in sys.argv[1:]) if v}))' "$@"; }
pretty() { if command -v jq >/dev/null; then jq "${1:-.}"; else python3 -m json.tool; fi; }
call() {  # prints the body on success; on HTTP errors prints the structured error to stderr and exits 1
  local out code
  out="$(curl -sS -w $'\n%{http_code}' "${HDR[@]}" "$@")" || exit 1
  code="${out##*$'\n'}"; out="${out%$'\n'*}"
  if (( code >= 400 )); then echo "HTTP $code" >&2; printf '%s\n' "$out" | pretty >&2; exit 1; fi
  printf '%s\n' "$out"
}

cmd="${1:-help}"; shift || true
case "$cmd" in
  models)
    call "$URL/v1/models" | pretty '.[] | {name, provider, model_id, available, status, aliases}' ;;
  ask)
    call -X POST "$URL/v1/chat" -d "$(json "content=${1:?text}" "model=${2:-}")" \
      | pretty '{reply: .message.content, conversation_id, model_used: .model_used.name, fell_back, usage, latency_ms}' ;;
  say)
    call -X POST "$URL/v1/conversations/${1:?id}/messages" -d "$(json "content=${2:?text}" "model=${3:-}")" \
      | pretty '{reply: .message.content, model_used: .model_used.name, fell_back, usage, latency_ms}' ;;
  swap)
    call -X PATCH "$URL/v1/conversations/${1:?id}" -d "$(json "model=${2:?model}")" \
      | pretty '{id, model, messages: (.messages | length), model_history}' ;;
  stream)
    curl -sS -N "${HDR[@]}" -X POST "$URL/v1/chat" \
      -d "$(python3 -c 'import json,sys; print(json.dumps({"content": sys.argv[1], "model": sys.argv[2] or None, "stream": True}))' "${1:?text}" "${2:-}")" ;;
  history)
    call "$URL/v1/conversations/${1:?id}" | pretty '{id, model, model_history, messages: [.messages[] | {role, model: .model.name, content}]}' ;;
  *)
    sed -n '2,12p' "$0" ;;
esac
