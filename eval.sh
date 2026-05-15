#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
set -a && source .env && set +a

PYTHONUNBUFFERED=1 python main.py \
  --task babaisai \
  --eval-only \
  --agent-model gpt-5-nano \
  --agent-api-key "$AGENT_API_KEY" \
  --agent-base-url "$AGENT_BASE_URL" \
  2>&1 | tee logs/eval_$(date +%Y%m%d_%H%M%S).log
