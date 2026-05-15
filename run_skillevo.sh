#!/bin/bash

cd /home/leah/discoverskill || exit 1

set -a
source .env
set +a

mkdir -p logs

python main.py \
  --task babaisai \
  --iterations 10 \
  --invent-every 1 \
  --n-eval 10 \
  --agent-model gpt-5-mini \
  --invent-model gpt-5 \
  --agent-api-key "$AGENT_API_KEY" \
  --agent-base-url "$AGENT_BASE_URL" \
  --invent-api-key "$INVENT_API_KEY" \
  --invent-base-url "$INVENT_BASE_URL" \
  2>&1 | tee logs/train_$(date +%Y%m%d_%H%M%S).log