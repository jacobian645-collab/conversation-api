#!/usr/bin/env bash
# Fine-tune Amazon Nova on Bedrock with the JSONL that `convo export` writes. Run in AWS CloudShell.
#
#   bash scripts/bedrock_finetune_nova.sh data/nova/train.jsonl [data/nova/validation.jsonl]
#   bash scripts/bedrock_finetune_nova.sh status <job-name>      # check a job
#
# What it does: validates the files against Nova's rules, checks the base model can be fine-tuned in your
# region, uploads to a private S3 bucket, creates (once) an IAM service role Bedrock can assume to read that
# bucket, and starts a model customization job.
#
# Env (all optional):
#   BASE_MODEL      default amazon.nova-lite-v1:0:300k (the script lists what your region offers if wrong)
#   AWS_REGION      default us-east-1
#   BUCKET          default conversation-api-finetune-<account>-<region>
#   ROLE_NAME       default ConversationApiBedrockFineTuning
#   EPOCHS=2  LEARNING_RATE=0.00001  WARMUP_STEPS=<auto: samples / 640 micro, 160 lite, 320 pro; max 100>
#   CUSTOM_MODEL_NAME, JOB_NAME    default convo-nova-<utc timestamp>
#   DRY_RUN=1       do everything except start the job; print the command instead
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"   # paths given as arguments stay relative to where you run it

REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}"

if [[ "${1:-}" == "status" ]]; then
  aws bedrock get-model-customization-job --region "$REGION" --job-identifier "${2:?job name}" \
    --query '{status: status, failure: failureMessage, customModel: outputModelArn, started: creationTime, ended: endTime}'
  exit 0
fi

TRAIN="${1:?usage: bash scripts/bedrock_finetune_nova.sh train.jsonl [validation.jsonl]}"
VAL="${2:-}"
[[ -f "$TRAIN" ]] || { echo "no such file: $TRAIN"; exit 1; }
[[ -z "$VAL" || -f "$VAL" ]] || { echo "no such file: $VAL"; exit 1; }

BASE_MODEL="${BASE_MODEL:-amazon.nova-lite-v1:0:300k}"
ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
BUCKET="${BUCKET:-conversation-api-finetune-${ACCOUNT}-${REGION}}"
ROLE_NAME="${ROLE_NAME:-ConversationApiBedrockFineTuning}"
STAMP="$(date -u +%Y%m%d-%H%M%S)"
JOB="${JOB_NAME:-convo-nova-${STAMP}}"
MODEL_NAME="${CUSTOM_MODEL_NAME:-convo-nova-${STAMP}}"
PREFIX="fine-tuning/${JOB}"
EPOCHS="${EPOCHS:-2}"
LEARNING_RATE="${LEARNING_RATE:-0.00001}"

# ---- 1. validate ---------------------------------------------------------------------------------------
CONVO="$(command -v convo || true)"; [[ -z "$CONVO" && -x "$ROOT/.venv/bin/convo" ]] && CONVO="$ROOT/.venv/bin/convo"
if [[ -n "$CONVO" ]]; then
  echo "▸ validating against Nova's rules"
  "$CONVO" validate-jsonl "$TRAIN" ${VAL:+--validation "$VAL"}
else
  echo "▸ convo not installed here; skipping local validation"
fi
SAMPLES="$(grep -c . "$TRAIN")"

# ---- 2. base model -----------------------------------------------------------------------------------
echo "▸ checking $BASE_MODEL can be fine-tuned in $REGION"
AVAILABLE="$(aws bedrock list-foundation-models --region "$REGION" --by-customization-type FINE_TUNING \
  --query "modelSummaries[?contains(modelId, 'nova')].modelId" --output text | tr '\t' '\n')"
if ! grep -qxF "$BASE_MODEL" <<<"$AVAILABLE"; then
  echo "✗ $BASE_MODEL is not fine-tunable in $REGION. Nova models you can fine-tune here:"
  echo "${AVAILABLE:-  (none: try AWS_REGION=us-east-1)}" | sed 's/^/  /'
  echo "Re-run with BASE_MODEL=<one of the above>"
  exit 1
fi

case "$BASE_MODEL" in *micro*) DIV=640 ;; *pro*) DIV=320 ;; *) DIV=160 ;; esac
WARMUP="${WARMUP_STEPS:-$((SAMPLES / DIV))}"
(( WARMUP > 100 )) && WARMUP=100

# ---- 3. private S3 bucket + upload ---------------------------------------------------------------------
if ! aws s3api head-bucket --bucket "$BUCKET" >/dev/null 2>&1; then
  echo "▸ creating bucket s3://$BUCKET"
  if [[ "$REGION" == "us-east-1" ]]; then
    aws s3api create-bucket --bucket "$BUCKET" --region "$REGION" >/dev/null
  else
    aws s3api create-bucket --bucket "$BUCKET" --region "$REGION" \
      --create-bucket-configuration "LocationConstraint=$REGION" >/dev/null
  fi
  aws s3api put-public-access-block --bucket "$BUCKET" --public-access-block-configuration \
    BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
fi
echo "▸ uploading to s3://$BUCKET/$PREFIX/"
aws s3 cp "$TRAIN" "s3://$BUCKET/$PREFIX/train.jsonl" --only-show-errors
[[ -n "$VAL" ]] && aws s3 cp "$VAL" "s3://$BUCKET/$PREFIX/validation.jsonl" --only-show-errors

# ---- 4. service role Bedrock assumes to read the data and write results ---------------------------------
TRUST=$(cat <<EOF
{"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Principal": {"Service": "bedrock.amazonaws.com"},
 "Action": "sts:AssumeRole", "Condition": {"StringEquals": {"aws:SourceAccount": "$ACCOUNT"},
 "ArnLike": {"aws:SourceArn": "arn:aws:bedrock:$REGION:$ACCOUNT:model-customization-job/*"}}}]}
EOF
)
ACCESS=$(cat <<EOF
{"Version": "2012-10-17", "Statement": [
 {"Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject"], "Resource": "arn:aws:s3:::$BUCKET/*"},
 {"Effect": "Allow", "Action": ["s3:ListBucket"], "Resource": "arn:aws:s3:::$BUCKET"}]}
EOF
)
NEW_ROLE=""
if ! aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1; then
  echo "▸ creating IAM role $ROLE_NAME"
  aws iam create-role --role-name "$ROLE_NAME" --assume-role-policy-document "$TRUST" \
    --description "Bedrock model customization for conversation-api" >/dev/null
  NEW_ROLE=1
fi
aws iam put-role-policy --role-name "$ROLE_NAME" --policy-name "s3-${BUCKET}" --policy-document "$ACCESS"
ROLE_ARN="$(aws iam get-role --role-name "$ROLE_NAME" --query Role.Arn --output text)"
if [[ -n "$NEW_ROLE" && -z "${DRY_RUN:-}" ]]; then echo "▸ waiting for the new role to propagate"; sleep 15; fi

# ---- 5. start the job ----------------------------------------------------------------------------------
TRAINING="{\"s3Uri\": \"s3://$BUCKET/$PREFIX/train.jsonl\"}"
OUTPUT="{\"s3Uri\": \"s3://$BUCKET/$PREFIX/output/\"}"
HYPER="{\"epochCount\": \"$EPOCHS\", \"learningRate\": \"$LEARNING_RATE\", \"learningRateWarmupSteps\": \"$WARMUP\"}"
CMD=(aws bedrock create-model-customization-job --region "$REGION"
  --job-name "$JOB" --custom-model-name "$MODEL_NAME" --role-arn "$ROLE_ARN"
  --base-model-identifier "$BASE_MODEL" --customization-type FINE_TUNING
  --training-data-config "$TRAINING" --output-data-config "$OUTPUT" --hyper-parameters "$HYPER")
if [[ -n "$VAL" ]]; then
  CMD+=(--validation-data-config "{\"validators\": [{\"s3Uri\": \"s3://$BUCKET/$PREFIX/validation.jsonl\"}]}")
fi

echo "▸ job: $JOB · base: $BASE_MODEL · samples: $SAMPLES · epochs $EPOCHS · lr $LEARNING_RATE · warmup $WARMUP"
if [[ -n "${DRY_RUN:-}" ]]; then
  echo "DRY_RUN: not starting. Command:"; printf ' %q' "${CMD[@]}"; echo; exit 0
fi
JOB_ARN="$("${CMD[@]}" --query jobArn --output text)"

cat <<EOF

✔ fine-tuning job started: $JOB_ARN

Check progress (training typically takes from tens of minutes to hours):
  bash scripts/bedrock_finetune_nova.sh status $JOB

When status is Completed, serve the custom model on demand and point the API at it:
  MODEL_ARN=\$(aws bedrock get-model-customization-job --region $REGION --job-identifier $JOB --query outputModelArn --output text)
  DEPLOY_ARN=\$(aws bedrock create-custom-model-deployment --region $REGION --model-deployment-name $MODEL_NAME \\
      --model-arn "\$MODEL_ARN" --query customModelDeploymentArn --output text)
  convo ask "hi" -m "bedrock:\$DEPLOY_ARN"                      # try it
  aws ssm put-parameter --name /conversation-api/models/nova-custom --type String --value "\$DEPLOY_ARN" --overwrite
  # then use ssm:/conversation-api/models/nova-custom in config/models.yaml (or an env var) and run: convo reload

Training metrics (loss per step) are written to s3://$BUCKET/$PREFIX/output/
EOF
