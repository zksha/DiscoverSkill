"""
Open-ended Skill Invention.

The inventor reflects on the most recent eval group:
  - which skills were in the group
  - each skill's contribution C (usage × rank)
  - the group's reward
  - the task that was attempted

It uses this to invent new skills that fix failures, strengthen successes,
or explore directions the current pool is missing.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

import openai

logger = logging.getLogger(__name__)

_CLIENT: Optional[openai.OpenAI] = None


def _client() -> openai.OpenAI:
    global _CLIENT
    if _CLIENT is None:
        key = os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY", "")
        _CLIENT = openai.OpenAI(api_key=key)
    return _CLIENT


_INVENT_PROMPT = """\
You are an AI research scientist designing cognitive skills for an AI agent.

{env_description}
## Last evaluated group

Reward: {last_reward:.2f} (mean across all episodes; 1.0 = fully correct, 0.0 = fully incorrect)

Skills in the group with their post-update values and contributions \
(C = usage × quality rank; V = updated EMA value; ★ = high contribution):

{group_detail}

## Episode outcomes (all episodes)

Each line shows the task attempted and the reward the agent received:

{last_task}

## Episode traces (3 randomly sampled episodes — full reasoning showcase)

{agent_trace}

## Full skill pool (for context — avoid duplicates)
{pool_summary}

---

Study the episode outcomes and traces carefully:
- Which tasks were solved (reward=1.0) and which failed (reward=0.0)?
- In the detailed traces, where did the reasoning go wrong or miss a step?
- Which skills were applied well (★) and which were ignored (○)?
- What capability is absent from the current pool that would have helped across the failures?

Then invent exactly {n_new} new skills to address the gaps you found.

Good skills are:
- Specific and actionable (not vague platitudes)
- Distinct from existing skills in the pool
- Directly motivated by the failure patterns in the trace above

Respond with a JSON array of exactly {n_new} objects:
[
  {{
    "name": "<snake_case_name>",
    "description": "<one sentence, ≤15 words, starting with a verb — this is the skill menu index the agent sees before deciding whether to invoke>",
    "content": "<2-5 paragraph markdown body explaining the strategy in detail>"
  }},
  ...
]

The description is critical: the agent reads ONLY this line when deciding whether to invoke the skill.
It must be specific enough that the agent can tell at a glance what the skill does and when to use it.

Examples of good descriptions:
  "Decompose the problem into independent sub-questions before solving each."
  "Retrieve and list relevant domain facts before starting to reason."
  "Check the final answer against the original question for consistency."

Respond with the JSON array only — no explanation outside it.
"""


def _format_group(group_detail: List[Dict[str, Any]]) -> str:
    lines = []
    for s in group_detail:
        C = s["contribution"]
        tag = "★" if C >= 0.5 else ("·" if C > 0 else "○")
        lines.append(
            f"  {tag} [{s['id']}] {s['name']}  C={C:.2f}  V={s['V']:.3f}\n"
            f"      {s['description']}"
        )
    return "\n".join(lines)


def invent_skills(
    pool_summary: str,
    last_group: List[Dict[str, Any]],
    last_reward: float,
    last_task: str,
    agent_trace: str,
    n_new: int = 3,
    model: str = "claude-haiku-4-5-20251001",
    client: Optional[openai.OpenAI] = None,
    env_description: str = "",
) -> List[Dict[str, str]]:
    """
    Invent n_new skills by reflecting on the last eval group + agent trace.

    Returns a list of dicts with keys: name, description, content.
    Returns [] on parse failure.
    """
    env_block = (env_description.strip() + "\n\n") if env_description.strip() else ""
    prompt = _INVENT_PROMPT.format(
        env_description=env_block,
        last_task=last_task,
        last_reward=last_reward,
        group_detail=_format_group(last_group),
        agent_trace=agent_trace,
        pool_summary=pool_summary,
        n_new=n_new,
    )

    response = (client or _client()).chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = (response.choices[0].message.content or "").strip()

    def _extract_array(text: str) -> list:
        # Try direct parse first (LLM usually returns clean JSON)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Fall back: find the JSON array by bracket counting
        start = text.find("[")
        if start == -1:
            return []
        depth, end = 0, -1
        for i, ch in enumerate(text[start:], start):
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end == -1:
            return []
        return json.loads(text[start : end + 1])

    try:
        skills = _extract_array(raw)
        validated = []
        for item in skills:
            if isinstance(item, dict) and "name" in item and "content" in item:
                validated.append({
                    "name": str(item["name"]),
                    "description": str(item.get("description", "")),
                    "content": str(item["content"]),
                })
        return validated[:n_new]
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        logger.warning("Invention parse error: %s | raw[:300]=%s", e, raw[:300])
    else:
        if not validated:
            logger.warning("Invention returned 0 valid items | raw[:300]=%s", raw[:300])
    return []
