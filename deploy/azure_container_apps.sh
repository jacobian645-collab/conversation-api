#!/usr/bin/env bash
# Deploy to Azure Container Apps from Azure Cloud Shell, Azure OpenAI as the default model, keyless.
#
#   export AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com
#   export AZURE_OPENAI_DEPLOYMENT=<deployment-name>
#   export AZURE_OPENAI_RESOURCE_ID=$(az cognitiveservices account show -n <resource> -g <rg> --query id -o tsv)
#   ./deploy/azure_container_apps.sh
#
# The app reaches Azure OpenAI with its system-assigned managed identity (no Azure OpenAI key anywhere).
# Callers use the app's API key, kept as a Container Apps secret.
set -euo pipefail
cd "$(dirname "$0")/.."

RG="${RG:-rg-conversation-api}"
LOCATION="${LOCATION:-eastus}"
APP="${APP:-conversation-api}"
: "${AZURE_OPENAI_ENDPOINT:?set AZURE_OPENAI_ENDPOINT}"
: "${AZURE_OPENAI_DEPLOYMENT:?set AZURE_OPENAI_DEPLOYMENT}"
API_KEY="${CONVO_API_KEY:-$(openssl rand -hex 24)}"

echo "▸ resource group $RG · location $LOCATION · app $APP"
az extension add --name containerapp --upgrade --only-show-errors
az provider register --namespace Microsoft.App --wait --only-show-errors
az provider register --namespace Microsoft.OperationalInsights --wait --only-show-errors
az group create -n "$RG" -l "$LOCATION" -o none

# Build in ACR from source and create the app (environment, registry and Log Analytics are created if missing).
az containerapp up --name "$APP" --resource-group "$RG" --location "$LOCATION" --source . \
  --ingress external --target-port 8080 \
  --env-vars "CONVO_DEFAULT_MODEL=azure-gpt" "AZURE_OPENAI_ENDPOINT=${AZURE_OPENAI_ENDPOINT}" \
             "AZURE_OPENAI_DEPLOYMENT=${AZURE_OPENAI_DEPLOYMENT}"

# App API key as a secret, referenced by env var.
az containerapp secret set -n "$APP" -g "$RG" --secrets "convo-api-key=${API_KEY}" -o none
az containerapp update -n "$APP" -g "$RG" --set-env-vars "CONVO_API_KEY=secretref:convo-api-key" \
  --min-replicas 0 --max-replicas 3 -o none

# Managed identity -> "Cognitive Services OpenAI User" on the Azure OpenAI resource.
PRINCIPAL="$(az containerapp identity assign -n "$APP" -g "$RG" --system-assigned --query principalId -o tsv)"
if [[ -n "${AZURE_OPENAI_RESOURCE_ID:-}" ]]; then
  az role assignment create --assignee-object-id "$PRINCIPAL" --assignee-principal-type ServicePrincipal \
    --role "Cognitive Services OpenAI User" --scope "$AZURE_OPENAI_RESOURCE_ID" -o none
else
  echo "! AZURE_OPENAI_RESOURCE_ID not set: grant 'Cognitive Services OpenAI User' to principal $PRINCIPAL yourself"
fi

FQDN="$(az containerapp show -n "$APP" -g "$RG" --query properties.configuration.ingress.fqdn -o tsv)"
cat <<EOF

✔ deployed: https://$FQDN    (role assignments can take a few minutes to apply)

Call it from Cloud Shell:
  export CONVO_URL=https://$FQDN
  export CONVO_API_KEY=$API_KEY
  convo ask "hello from Container Apps"     # or ./scripts/convo.sh ask "hello"
EOF
