"""
Semantic Credit Assignment — tag-grounded, ranking-based.

Two-stage process:
  1. Usage count  — parse [skill_id] tags from agent trace (objective)
  2. LLM ranking  — given the tagged trace, rank skills by contribution quality
                    (subjective, but grounded in explicit tag evidence)

Final score C(s_i) ∈ [0, 1]:
  C = usage_score × rank_score

  usage_score_i = count_i / max_count     (how often it was applied — the gate)
  rank_score_i  = (k − rank_i) / k       (rank 1 → 1.0, rank k → 1/k)

Multiplicative: a skill must be BOTH used frequently AND ranked highly to get
high credit. A skill never tagged by the agent gets C = 0 regardless of rank.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional

import openai

logger = logging.getLogger(__name__)

from .skill import Skill

_CLIENT: Optional[openai.OpenAI] = None


def _client() -> openai.OpenAI:
    global _CLIENT
    if _CLIENT is None:
        key = os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY", "")
        _CLIENT = openai.OpenAI(api_key=key)
    return _CLIENT


# --------------------------------------------------------------------------- #
# Stage 1: usage score from tag counts                                         #
# --------------------------------------------------------------------------- #

def _usage_scores(group: List[Skill], usage: Counter) -> Dict[str, float]:
    """Normalize tag counts to [0, 1]. Unused skills → 0."""
    counts = {s.id: usage.get(s.id, 0) for s in group}
    max_count = max(counts.values()) if counts else 1
    if max_count == 0:
        return {sid: 0.0 for sid in counts}
    return {sid: c / max_count for sid, c in counts.items()}


# --------------------------------------------------------------------------- #
# Stage 2: LLM ranking of tagged trace                                         #
# --------------------------------------------------------------------------- #

_RANK_PROMPT = """\
An AI agent solved a task by explicitly invoking skills on demand. \
Each skill was loaded and applied when the agent called INVOKE_SKILL.

## Task
{task_prompt}

## Skills invoked (with invocation counts)
{skill_list}

## Agent's reasoning trace
{trace}

## Outcome
Reward: {reward}  (1.0 = fully correct, 0.0 = fully incorrect)

---

Rank the invoked skills from most to least valuable for reaching the \
correct answer. Consider: did the skill's guidance visibly improve the \
reasoning? Did the agent apply it effectively after invoking it?

Respond with a JSON array of skill IDs, best first:
["skill_id_1", "skill_id_2", ...]

Include only skills that were invoked. Respond with the JSON array only.
"""


def _llm_rank(
    task_prompt: str,
    group: List[Skill],
    usage: Counter,
    trace_text: str,
    reward: float,
    model: str,
    client: Optional[anthropic.Anthropic] = None,
) -> Dict[str, float]:
    """
    Ask LLM to rank skills that were actually used.
    Returns rank_score ∈ [0, 1] per skill (0 for unused skills).
    """
    used_skills = [s for s in group if usage.get(s.id, 0) > 0]
    if not used_skills:
        return {s.id: 0.0 for s in group}

    skill_list = "\n".join(
        f"  [{s.id}] {s.name} — used {usage[s.id]}×" for s in used_skills
    )
    prompt = _RANK_PROMPT.format(
        task_prompt=task_prompt,
        skill_list=skill_list,
        trace=trace_text,
        reward=reward,
    )

    response = (client or _client()).chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = (response.choices[0].message.content or "").strip()

    ranked_ids: List[str] = []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("[")
        parsed = []
        if start != -1:
            depth, end = 0, -1
            for i, ch in enumerate(raw[start:], start):
                if ch == "[":
                    depth += 1
                elif ch == "]":
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            try:
                parsed = json.loads(raw[start : end + 1]) if end != -1 else []
            except (json.JSONDecodeError, TypeError):
                parsed = []
    if isinstance(parsed, list):
        ranked_ids = [x for x in parsed if isinstance(x, str)]

    # Convert rank → score: rank 1 → 1.0, rank k → 0.0
    valid_ids = {s.id for s in used_skills}
    ranked_ids = [sid for sid in ranked_ids if sid in valid_ids]
    k = len(ranked_ids)

    if not ranked_ids:
        logger.warning("Credit ranking returned no valid skill IDs | raw[:300]=%s", raw[:300])

    rank_scores: Dict[str, float] = {s.id: 0.0 for s in group}
    for i, sid in enumerate(ranked_ids):
        rank_scores[sid] = (k - i) / k   # rank 1 → 1.0, rank k → 1/k
    return rank_scores


# --------------------------------------------------------------------------- #
# Combined credit                                                               #
# --------------------------------------------------------------------------- #

def assign_credit(
    task_prompt: str,
    group: List[Skill],
    trace: List[Dict[str, Any]],
    usage: Counter,
    reward: float,
    model: str = "claude-haiku-4-5-20251001",
    agent_trace_text: Optional[str] = None,
    client: Optional[anthropic.Anthropic] = None,
) -> Dict[str, float]:
    """
    Compute C(s_i) ∈ [0, 1] for each skill.

    C = usage_score × rank_score

    usage_score gates on whether the skill was applied at all.
    rank_score weights by quality given the tagged trace.
    A skill never tagged by the agent gets C = 0 regardless of rank.

    agent_trace_text overrides automatic extraction from trace — useful for
    interactive episodes (MiniHack) where trace contains many short messages.
    """
    if agent_trace_text is None:
        agent_trace_text = next(
            (m["content"] for m in trace if m["role"] == "assistant"), ""
        )

    u_scores = _usage_scores(group, usage)
    r_scores = _llm_rank(task_prompt, group, usage, agent_trace_text, reward, model, client)

    return {
        s.id: u_scores[s.id] * r_scores[s.id]
        for s in group
    }
