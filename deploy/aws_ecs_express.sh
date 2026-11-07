#!/usr/bin/env bash
# Deploy the Conversation API to Amazon ECS Express Mode (Fargate behind an HTTPS load balancer) and, if you
# pass AMPLIFY_APP_ID, point the chat page on AWS Amplify Hosting at it. Run from AWS CloudShell in the repo.
#
#   AMPLIFY_APP_ID=d1abc2def3 ./deploy/aws_ecs_express.sh          # build, push, create or update, wire Amplify
#   ./deploy/aws_ecs_express.sh status                              # URL, health, Amplify setting
#   ./deploy/aws_ecs_express.sh key                                 # print the API key
#   ./deploy/aws_ecs_express.sh destroy                             # delete the service (keeps image, roles, key)
#
# What it creates (all named conversation-api*, region $AWS_REGION, default us-east-1):
#   ECR repository · CloudWatch log group (14 days) · SSM SecureString /conversation-api/api-key
#   IAM roles: -exec (pull image, read the key), -task (Bedrock invoke + read /conversation-api/* parameters),
#              -infra (lets ECS Express Mode create the load balancer, target group, security groups, scaling)
#   ECS Express Mode service: 1 task, 0.5 vCPU / 1 GB, health check /healthz, HTTPS URL *.ecs.<region>.on.aws
#
# One task on purpose: conversations live in the task's memory (CONVO_STORE=memory). Two tasks behind the
# load balancer would each hold different conversations. A redeploy starts a fresh task, so conversations
# reset; see ADR-0002 for a shared store.
#
# Options (environment):
#   AMPLIFY_APP_ID    Amplify app ID; sets CORS for https://<branch>.<app domain>, sets CONVO_API_URL, redeploys
#   AMPLIFY_BRANCH    default main
#   EXTRA_ORIGINS     more browser origins, comma-separated (e.g. a custom domain)
#   CONVO_DEFAULT_MODEL  default smart (Claude Sonnet on Bedrock); BEDROCK_SONNET_ID etc. are passed through,
#                     so BEDROCK_SONNET_ID=ssm:/conversation-api/models/sonnet works as it does in CloudShell
#   CPU / MEMORY      default 512 / 1024 (CPU units / MiB)
#   SERVICE_NAME      default conversation-api
#   SKIP_BUILD=1      reuse the last pushed image
set -euo pipefail

CMD=${1:-deploy}
REGION=${AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}
NAME=${SERVICE_NAME:-conversation-api}
CPU=${CPU:-512}
MEMORY=${MEMORY:-1024}
AMPLIFY_APP_ID=${AMPLIFY_APP_ID:-}
AMPLIFY_BRANCH=${AMPLIFY_BRANCH:-main}
EXTRA_ORIGINS=${EXTRA_ORIGINS:-}
DEFAULT_MODEL=${CONVO_DEFAULT_MODEL:-smart}
PARAM_PREFIX=/conversation-api
KEY_PARAM=$PARAM_PREFIX/api-key
LOG_GROUP=/ecs/$NAME
EXEC_ROLE=$NAME-exec
TASK_ROLE=$NAME-task
INFRA_ROLE=$NAME-infra
export AWS_REGION=$REGION AWS_DEFAULT_REGION=$REGION AWS_PAGER=""

say() { printf '\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33m%s\033[0m\n' "$*" >&2; }
die() { printf '\033[31merror: %s\033[0m\n' "$*" >&2; exit 1; }
py() { python3 -c "$@"; }

ACCOUNT=$(aws sts get-caller-identity --query Account --output text) || die "no AWS credentials in this shell"
SERVICE_ARN=arn:aws:ecs:$REGION:$ACCOUNT:service/default/$NAME
REPO_URI=$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/$NAME

need_express_cli() {
  aws ecs create-express-gateway-service help >/dev/null 2>&1 ||
    die "this AWS CLI has no ECS Express Mode commands; update it (CloudShell: sudo dnf update -y awscli2, or reinstall AWS CLI v2)"
}

service_json() { aws ecs describe-express-gateway-service --service-arn "$SERVICE_ARN" --output json 2>/dev/null || true; }

service_status() {  # ACTIVE | DRAINING | INACTIVE | (empty when there is none)
  service_json | py 'import json,sys
d=sys.stdin.read().strip()
print(json.loads(d)["service"]["status"]["statusCode"] if d else "")'
}

service_url() {  # https://... of the public endpoint, or empty while provisioning
  service_json | py 'import json,sys
d=sys.stdin.read().strip()
if not d: raise SystemExit
for c in json.loads(d)["service"].get("activeConfigurations") or []:
    for p in c.get("ingressPaths") or []:
        e = p.get("endpoint") or ""
        if e and p.get("accessType", "PUBLIC") == "PUBLIC":
            print(e if e.startswith("http") else "https://" + e.rstrip("/"))
            raise SystemExit'
}

amplify_origin() {
  local domain
  domain=$(aws amplify get-app --app-id "$AMPLIFY_APP_ID" --query app.defaultDomain --output text) ||
    die "no Amplify app $AMPLIFY_APP_ID in $REGION (create it in the Amplify console first, connected to your GitHub repo)"
  echo "https://$AMPLIFY_BRANCH.$domain"
}

role_arn() { aws iam get-role --role-name "$1" --query Role.Arn --output text 2>/dev/null || true; }

ensure_role() {  # name, service principal, [managed policy ARN]
  local arn
  arn=$(role_arn "$1")
  if [[ -z $arn ]]; then
    say "Creating IAM role $1" >&2
    arn=$(aws iam create-role --role-name "$1" --query Role.Arn --output text \
      --assume-role-policy-document "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Principal\":{\"Service\":\"$2\"},\"Action\":\"sts:AssumeRole\"}]}")
  fi
  [[ -n ${3:-} ]] && aws iam attach-role-policy --role-name "$1" --policy-arn "$3"
  echo "$arn"
}

ssm_params_for_models() {  # parameter ARNs referenced as ssm:/... by BEDROCK_*_ID variables
  local v
  for v in BEDROCK_SONNET_ID BEDROCK_HAIKU_ID BEDROCK_NOVA_ID BEDROCK_SMALL_MODEL_ID; do
    if [[ ${!v:-} == ssm:/* ]]; then echo "arn:aws:ssm:$REGION:$ACCOUNT:parameter/${!v#ssm:/}"; fi
  done
}

inherit_model_ids() {  # a redeploy keeps the running service's BEDROCK_* settings unless this shell sets them
  local k v
  while IFS='=' read -r k v; do
    if [[ -n $k && -z ${!k:-} ]]; then
      export "$k=$v"
      say "Keeping $k=$v from the running service"
    fi
  done < <(service_json | py 'import json,sys
d=sys.stdin.read().strip()
if not d: raise SystemExit
for c in (json.loads(d)["service"].get("activeConfigurations") or [])[:1]:
    for e in (c.get("primaryContainer") or {}).get("environment") or []:
        if e["name"].startswith("BEDROCK_") and e["name"].isidentifier():
            print(e["name"] + "=" + e["value"])')
}

deploy() {
  need_express_cli
  inherit_model_ids
  command -v docker >/dev/null || [[ ${SKIP_BUILD:-} == 1 ]] || die "docker not found (AWS CloudShell has it; or build elsewhere and use SKIP_BUILD=1)"

  # ---- who may call it from a browser --------------------------------------------------------------
  local origins="$EXTRA_ORIGINS" origin=""
  if [[ -n $AMPLIFY_APP_ID ]]; then
    origin=$(amplify_origin)
    origins=${origins:+$origins,}$origin
  fi
  [[ -n $origins ]] || warn "No AMPLIFY_APP_ID or EXTRA_ORIGINS: the API will refuse browser calls from other sites (its own /ui still works)."

  # ---- API key, kept in Parameter Store ------------------------------------------------------------
  if ! aws ssm get-parameter --name "$KEY_PARAM" >/dev/null 2>&1; then
    say "Creating API key in Parameter Store ($KEY_PARAM)"
    aws ssm put-parameter --name "$KEY_PARAM" --type SecureString \
      --value "$(py 'import secrets; print(secrets.token_urlsafe(32))')" \
      --description "CONVO_API_KEY for $NAME" >/dev/null
  fi
  local key_arn=arn:aws:ssm:$REGION:$ACCOUNT:parameter$KEY_PARAM

  # ---- IAM -----------------------------------------------------------------------------------------
  local exec_arn task_arn infra_arn new_roles=0 r
  for r in "$EXEC_ROLE" "$TASK_ROLE" "$INFRA_ROLE"; do [[ -n $(role_arn "$r") ]] || new_roles=1; done
  exec_arn=$(ensure_role "$EXEC_ROLE" ecs-tasks.amazonaws.com arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy)
  aws iam put-role-policy --role-name "$EXEC_ROLE" --policy-name read-api-key --policy-document \
    "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Action\":\"ssm:GetParameters\",\"Resource\":\"$key_arn\"}]}"
  task_arn=$(ensure_role "$TASK_ROLE" ecs-tasks.amazonaws.com)
  local extra_params
  extra_params=$(ssm_params_for_models | py 'import json,sys; print(json.dumps([l.strip() for l in sys.stdin if l.strip()]))')
  aws iam put-role-policy --role-name "$TASK_ROLE" --policy-name bedrock-and-model-ids --policy-document "$(
    ACCOUNT=$ACCOUNT REGION=$REGION PREFIX=$PARAM_PREFIX EXTRA="$extra_params" py 'import json,os
a,r,p=os.environ["ACCOUNT"],os.environ["REGION"],os.environ["PREFIX"]
params=[f"arn:aws:ssm:{r}:{a}:parameter{p}/models/*"]+json.loads(os.environ["EXTRA"])
print(json.dumps({"Version":"2012-10-17","Statement":[
 {"Sid":"InvokeBedrock","Effect":"Allow",
  "Action":["bedrock:InvokeModel","bedrock:InvokeModelWithResponseStream"],
  "Resource":["arn:aws:bedrock:*::foundation-model/*",
              f"arn:aws:bedrock:*:{a}:inference-profile/*",
              f"arn:aws:bedrock:*:{a}:custom-model-deployment/*",
              f"arn:aws:bedrock:*:{a}:provisioned-model/*"]},
 {"Sid":"ReadModelIds","Effect":"Allow","Action":["ssm:GetParameter","ssm:GetParameters"],"Resource":sorted(set(params))}]}))')"
  infra_arn=$(ensure_role "$INFRA_ROLE" ecs.amazonaws.com arn:aws:iam::aws:policy/service-role/AmazonECSInfrastructureRoleforExpressGatewayServices)
  if [[ $new_roles == 1 ]]; then say "Waiting 20 s for new IAM roles to propagate"; sleep 20; fi

  # ---- logs ----------------------------------------------------------------------------------------
  if ! aws logs describe-log-groups --log-group-name-prefix "$LOG_GROUP" --query "logGroups[?logGroupName=='$LOG_GROUP'] | length(@)" --output text | grep -q '^1$'; then
    aws logs create-log-group --log-group-name "$LOG_GROUP"
    aws logs put-retention-policy --log-group-name "$LOG_GROUP" --retention-in-days 14
  fi

  # ---- image ---------------------------------------------------------------------------------------
  aws ecr describe-repositories --repository-names "$NAME" >/dev/null 2>&1 ||
    aws ecr create-repository --repository-name "$NAME" --image-scanning-configuration scanOnPush=true >/dev/null
  local tag image
  if [[ ${SKIP_BUILD:-} == 1 ]]; then
    tag=$(aws ecr describe-images --repository-name "$NAME" --query 'sort_by(imageDetails,&imagePushedAt)[-1].imageTags[0]' --output text)
    [[ -n $tag && $tag != None ]] || die "SKIP_BUILD=1 but the ECR repository has no tagged image"
  else
    tag=$(git rev-parse --short HEAD 2>/dev/null || echo build)-$(date +%Y%m%d%H%M%S)
    say "Building $NAME:$tag (linux/amd64)"
    docker build --platform linux/amd64 -t "$REPO_URI:$tag" .
    aws ecr get-login-password | docker login --username AWS --password-stdin "${REPO_URI%%/*}" >/dev/null
    docker push "$REPO_URI:$tag"
  fi
  image=$REPO_URI:$tag

  # ---- container definition ------------------------------------------------------------------------
  local container
  container=$(IMAGE=$image ORIGINS=$origins MODEL=$DEFAULT_MODEL LOG_GROUP=$LOG_GROUP KEY_ARN=$key_arn py 'import json,os
e=os.environ
env={"CONVO_CORS_ORIGINS":e["ORIGINS"],"CONVO_DEFAULT_MODEL":e["MODEL"],"CONVO_STORE":"memory",
     "AWS_REGION":e["AWS_REGION"],"CONVO_LOG_LEVEL":"info"}
for k in ("BEDROCK_SONNET_ID","BEDROCK_HAIKU_ID","BEDROCK_NOVA_ID","BEDROCK_SMALL_MODEL_ID","BEDROCK_MAX_ATTEMPTS"):
    if e.get(k): env[k]=e[k]
print(json.dumps({"image":e["IMAGE"],"containerPort":8080,
  "awsLogsConfiguration":{"logGroup":e["LOG_GROUP"],"logStreamPrefix":"api"},
  "environment":[{"name":k,"value":v} for k,v in env.items()],
  "secrets":[{"name":"CONVO_API_KEY","valueFrom":e["KEY_ARN"]}]}))')
  local scaling='{"minTaskCount":1,"maxTaskCount":1}'

  # ---- create or update ----------------------------------------------------------------------------
  local status
  status=$(service_status)
  if [[ $status == ACTIVE ]]; then
    say "Updating $NAME (rolling deployment)"
    aws ecs update-express-gateway-service --service-arn "$SERVICE_ARN" \
      --primary-container "$container" --task-role-arn "$task_arn" \
      --cpu "$CPU" --memory "$MEMORY" --health-check-path /healthz --scaling-target "$scaling" >/dev/null
  else
    [[ $status == DRAINING ]] && die "$NAME is still being deleted; try again in a few minutes"
    say "Creating ECS Express Mode service $NAME"
    local attempt
    for attempt in 1 2 3 4 5 6; do
      if aws ecs create-express-gateway-service --service-name "$NAME" \
        --execution-role-arn "$exec_arn" --infrastructure-role-arn "$infra_arn" --task-role-arn "$task_arn" \
        --primary-container "$container" --cpu "$CPU" --memory "$MEMORY" \
        --health-check-path /healthz --scaling-target "$scaling" \
        --tags key=app,value="$NAME" >/dev/null 2>/tmp/convo-express.err; then
        break
      fi
      if grep -qiE 'assume|role|not authorized to perform: iam:PassRole|propagat' /tmp/convo-express.err && [[ $attempt -lt 6 ]]; then
        warn "IAM not ready yet (attempt $attempt): $(head -c 300 /tmp/convo-express.err)"; sleep 20
      else
        cat /tmp/convo-express.err >&2; die "could not create the service"
      fi
    done
  fi

  # ---- wait until it answers -----------------------------------------------------------------------
  say "Waiting for the service and its HTTPS endpoint (first deploy takes about 5-10 minutes)"
  local url=""
  for _ in $(seq 1 90); do
    url=$(service_url)
    if [[ -n $url ]] && curl -fsS -m 5 "$url/healthz" >/dev/null 2>&1; then break; fi
    sleep 10
  done
  [[ -n $url ]] || die "no endpoint yet; run '$0 status' in a few minutes"
  curl -fsS -m 5 "$url/healthz" >/dev/null || die "endpoint $url is not healthy yet; check logs: aws logs tail $LOG_GROUP --follow"
  say "API is up: $url"

  # ---- point the Amplify page at it ----------------------------------------------------------------
  if [[ -n $AMPLIFY_APP_ID ]]; then
    say "Setting CONVO_API_URL on Amplify app $AMPLIFY_APP_ID and redeploying $AMPLIFY_BRANCH"
    local vars
    vars=$(aws amplify get-app --app-id "$AMPLIFY_APP_ID" --query app.environmentVariables --output json |
      URL=$url py 'import json,os,sys
v=json.load(sys.stdin) or {}
v["CONVO_API_URL"]=os.environ["URL"]
print(json.dumps(v))')
    aws amplify update-app --app-id "$AMPLIFY_APP_ID" --environment-variables "$vars" >/dev/null
    aws amplify start-job --app-id "$AMPLIFY_APP_ID" --branch-name "$AMPLIFY_BRANCH" --job-type RELEASE >/dev/null
  fi

  summary "$url" "$origin"
}

summary() {
  local url=$1 origin=${2:-}
  cat <<EOF

  API          $url        (its own chat page: $url/ui)
EOF
  [[ -n $origin ]] && echo "  Chat page    $origin   (Amplify is rebuilding it now; about a minute)"
  cat <<EOF
  API key      ./deploy/aws_ecs_express.sh key
  Logs         aws logs tail $LOG_GROUP --follow
  Smoke test   curl -s $url/v1/models -H "X-API-Key: \$(./deploy/aws_ecs_express.sh key)" | head -c 300

EOF
}

status() {
  need_express_cli
  local s url
  s=$(service_status)
  [[ -n $s ]] || { echo "No service $NAME in $REGION."; return; }
  url=$(service_url)
  echo "Service   $NAME: $s"
  echo "API       ${url:-provisioning}"
  [[ -n $url ]] && echo "Health    $(curl -fsS -m 5 "$url/healthz" 2>&1 || echo 'not answering yet')"
  if [[ -n $AMPLIFY_APP_ID ]]; then
    echo "Amplify   CONVO_API_URL=$(aws amplify get-app --app-id "$AMPLIFY_APP_ID" --query 'app.environmentVariables.CONVO_API_URL' --output text)"
    echo "Chat page $(amplify_origin)"
  fi
}

case $CMD in
  deploy) deploy ;;
  status) status ;;
  key) aws ssm get-parameter --name "$KEY_PARAM" --with-decryption --query Parameter.Value --output text ;;
  destroy)
    need_express_cli
    say "Deleting ECS Express Mode service $NAME (the load balancer and tasks go with it)"
    aws ecs delete-express-gateway-service --service-arn "$SERVICE_ARN" >/dev/null
    cat <<EOF
Kept, in case you deploy again (delete by hand if you're done):
  aws ecr delete-repository --repository-name $NAME --force
  aws ssm delete-parameter --name $KEY_PARAM
  aws logs delete-log-group --log-group-name $LOG_GROUP
  IAM roles $EXEC_ROLE, $TASK_ROLE, $INFRA_ROLE
EOF
    ;;
  *) die "usage: $0 [deploy|status|key|destroy]" ;;
esac
