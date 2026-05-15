#!/usr/bin/env bash
# BabaIsYou skill evolution — training script
#
# Time estimate (claude-haiku-4-5, 10 parallel episodes, max_steps=30):
#   ~25-30s per iteration
#   30 iter ≈ 13-15 min
#   50 iter ≈ 20-25 min
#
# Usage:
#   ./train.sh                       # defaults below
#   ITERATIONS=50 ./train.sh         # override iterations
#   AGENT_API_KEY=sk-... ./train.sh  # separate key for agent

set -euo pipefail

# ── Load .env if present ───────────────────────────────────────────────────────
if [[ -f "$(dirname "$0")/.env" ]]; then
    set -o allexport
    # shellcheck disable=SC1091
    source "$(dirname "$0")/.env"
    set +o allexport
fi

# ── Parameters (override via env) ─────────────────────────────────────────────
ITERATIONS=${ITERATIONS:-30}
GROUP_SIZE=${GROUP_SIZE:-4}
N_EVAL=${N_EVAL:-10}         # episodes per iter (= n_train, so covers all train tasks)
N_TRAIN=${N_TRAIN:-10}       # training task count (rest become test set)
MAX_STEPS=${MAX_STEPS:-30}   # game steps per episode (shorter = faster iterations)
INVENT_EVERY=${INVENT_EVERY:-2}
N_NEW=${N_NEW:-3}
ALPHA=${ALPHA:-0.1}
TAU=${TAU:-2.0}
SEED=${SEED:-42}
MODEL=${MODEL:-gpt-4o-mini}
AGENT_MODEL=${AGENT_MODEL:-}    # overrides MODEL for agent episodes
INVENT_MODEL=${INVENT_MODEL:-}  # overrides MODEL for inventor + credit
REPO=${REPO:-skill_repo}

# API keys: fall back to ANTHROPIC_API_KEY if not set separately
AGENT_API_KEY=${AGENT_API_KEY:-}
AGENT_BASE_URL=${AGENT_BASE_URL:-}
INVENT_API_KEY=${INVENT_API_KEY:-}
INVENT_BASE_URL=${INVENT_BASE_URL:-}

# ── Validation ─────────────────────────────────────────────────────────────────
if [[ -z "${ANTHROPIC_API_KEY:-}" && -z "$AGENT_API_KEY" ]]; then
    echo "Error: set ANTHROPIC_API_KEY (or AGENT_API_KEY + INVENT_API_KEY)" >&2
    exit 1
fi

# ── Log setup ──────────────────────────────────────────────────────────────────
mkdir -p logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="logs/train_${TIMESTAMP}.log"

# ── Summary ────────────────────────────────────────────────────────────────────
echo "=============================="
echo " BabaIsYou Skill Evolution"
echo "=============================="
echo " iterations  : $ITERATIONS"
echo " group_size  : $GROUP_SIZE"
echo " n_eval      : $N_EVAL  (episodes/iter)"
echo " n_train     : $N_TRAIN  (train tasks)"
echo " max_steps   : $MAX_STEPS"
echo " invent_every: $INVENT_EVERY"
echo " n_new       : $N_NEW"
echo " model       : $MODEL"
echo " repo        : $REPO"
echo " log         : $LOGFILE"
echo "------------------------------"
EST_SECS=$(( ITERATIONS * 28 ))
echo " estimated   : ~$(( EST_SECS / 60 ))m $(( EST_SECS % 60 ))s"
echo "=============================="
echo ""

# ── Build argument list ────────────────────────────────────────────────────────
ARGS=(
    --task babaisai
    --iterations "$ITERATIONS"
    --group-size "$GROUP_SIZE"
    --n-eval "$N_EVAL"
    --n-train "$N_TRAIN"
    --max-steps "$MAX_STEPS"
    --invent-every "$INVENT_EVERY"
    --n-new "$N_NEW"
    --alpha "$ALPHA"
    --tau "$TAU"
    --seed "$SEED"
    --model "$MODEL"
    --repo "$REPO"
)

[[ -n "$AGENT_MODEL"  ]] && ARGS+=(--agent-model  "$AGENT_MODEL")
[[ -n "$INVENT_MODEL" ]] && ARGS+=(--invent-model "$INVENT_MODEL")

[[ -n "$AGENT_API_KEY"   ]] && ARGS+=(--agent-api-key   "$AGENT_API_KEY")
[[ -n "$AGENT_BASE_URL"  ]] && ARGS+=(--agent-base-url  "$AGENT_BASE_URL")
[[ -n "$INVENT_API_KEY"  ]] && ARGS+=(--invent-api-key  "$INVENT_API_KEY")
[[ -n "$INVENT_BASE_URL" ]] && ARGS+=(--invent-base-url "$INVENT_BASE_URL")

# ── Run ────────────────────────────────────────────────────────────────────────
python main.py "${ARGS[@]}" 2>&1 | tee "$LOGFILE"
