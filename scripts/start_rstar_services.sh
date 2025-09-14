#!/usr/bin/env bash
set -euo pipefail

# start_rstar_services.sh
# One-shot launcher for rStar tool infrastructure using Docker Compose:
# - Redis (backend queue)
# - Code Judge API (executes Python tool calls)
# - Code Judge Workers
# - vLLM OpenAI-compatible server (serves your model)
#
# Prerequisites:
# - Docker and Docker Compose installed
# - NVIDIA Container Toolkit if using GPU for vLLM
# - A local model directory mounted for vLLM via $MODEL_PATH (see .env.rstar)
#
# Usage:
#   ./scripts/start_rstar_services.sh  # will create .env.rstar if missing

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="$ROOT_DIR/docker-compose.rstar.yml"
ENV_FILE="$ROOT_DIR/.env.rstar"

if [[ ! -f "$COMPOSE_FILE" ]]; then
  echo "[ERROR] Missing $COMPOSE_FILE. Did you move or rename it?" >&2
  exit 1
fi

if [[ ! -f "$ENV_FILE" ]]; then
  cat > "$ENV_FILE" << 'EOF'
# .env.rstar - environment overrides for rStar services

# Redis location (internal to compose network)
REDIS_URI=redis://redis:6379

# Code Judge configuration
MAX_EXECUTION_TIME=4
MAX_WORKERS=64
CODE_JUDGE_PORT=8088

# vLLM configuration
VLLM_HOST=0.0.0.0
VLLM_PORT=8000
# Set MODEL_PATH to a local directory containing your model weights
MODEL_PATH=/models/gemma-3-12b-it

EOF
  echo "[INFO] Created default $ENV_FILE. Please review and adjust MODEL_PATH as needed."
fi

set -a
source "$ENV_FILE"
set +a

# Validate required variables
missing=()
[[ -z "${REDIS_URI:-}" ]] && missing+=(REDIS_URI)
[[ -z "${MAX_EXECUTION_TIME:-}" ]] && missing+=(MAX_EXECUTION_TIME)
[[ -z "${MAX_WORKERS:-}" ]] && missing+=(MAX_WORKERS)
[[ -z "${CODE_JUDGE_PORT:-}" ]] && missing+=(CODE_JUDGE_PORT)
[[ -z "${VLLM_HOST:-}" ]] && missing+=(VLLM_HOST)
[[ -z "${VLLM_PORT:-}" ]] && missing+=(VLLM_PORT)
[[ -z "${MODEL_PATH:-}" ]] && missing+=(MODEL_PATH)

if (( ${#missing[@]} > 0 )); then
  echo "[ERROR] Missing required variables in $ENV_FILE: ${missing[*]}" >&2
  echo "[HINT] Edit $ENV_FILE and set the values. Example:"
  echo "       MODEL_PATH=/abs/path/to/your/model  # must exist on host"
  echo "       VLLM_HOST=0.0.0.0"
  echo "       VLLM_PORT=8000"
  echo "       CODE_JUDGE_PORT=8088"
  echo "       REDIS_URI=redis://redis:6379"
  echo "       MAX_EXECUTION_TIME=4"
  echo "       MAX_WORKERS=64"
  exit 1
fi

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "[ERROR] MODEL_PATH directory does not exist on host: $MODEL_PATH" >&2
  exit 1
fi

echo "[INFO] Starting rStar services via docker compose..."
docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" up -d --build
echo "[INFO] Services started. Status:"
docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" ps

echo "[HINT] Verify vLLM server: curl http://localhost:$(grep VLLM_PORT $ENV_FILE | cut -d= -f2)/v1/models"
echo "[HINT] Code Judge API should be at http://localhost:$(grep CODE_JUDGE_PORT $ENV_FILE | cut -d= -f2)"
