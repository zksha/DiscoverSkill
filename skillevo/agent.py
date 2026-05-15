"""
Task-agnostic agent runner with on-demand skill invocation.

Flow:
  1. System prompt contains a skill menu — one-line descriptions only.
  2. Agent reasons freely; when it needs a skill it writes:
       INVOKE_SKILL: <skill_id>
  3. We detect the invocation, load the full skill .md from the eval
     workspace (skills/), and inject it as the next user turn.
  4. Agent continues reasoning with the full skill content in context.
  5. Loop until the agent writes "FINAL ANSWER: ..." or max_turns reached.

Credit assignment is trivial: count INVOKE_SKILL calls per skill_id.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import openai

from .skill import Skill
from .tasks.base import TaskInstance

_CLIENT: Optional[openai.OpenAI] = None
_INVOKE_RE = re.compile(r"INVOKE_SKILL\s*:\s*([a-z0-9_]+)", re.IGNORECASE)


def _client() -> openai.OpenAI:
    global _CLIENT
    if _CLIENT is None:
        key = os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY", "")
        _CLIENT = openai.OpenAI(api_key=key)
    return _CLIENT


# --------------------------------------------------------------------------- #
# Prompt construction                                                          #
# --------------------------------------------------------------------------- #

_SYSTEM_TMPL = """\
You are a capable AI agent. Your goal is to solve the given task as accurately \
as possible.

You have access to the following skills. Each skill provides detailed reasoning \
guidance. To use a skill, write on its own line:

  INVOKE_SKILL: <skill_id>

You will immediately receive the skill's full instructions and can apply them \
before continuing.

## Available skills

{skill_menu}

---

You may invoke skills in any order, as many times as needed. When you have \
reached a final answer, end your response with exactly:

FINAL ANSWER: <your answer>
"""


def _build_system(group: List[Skill]) -> str:
    menu = "\n".join(
        f"  [{s.id}] {s.description}" for s in group
    )
    return _SYSTEM_TMPL.format(skill_menu=menu)


def _load_skill_content(skill_id: str, eval_dir: str = "skills") -> Optional[str]:
    """Load full skill content from the eval workspace."""
    path = Path(eval_dir) / f"{skill_id}.md"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


# --------------------------------------------------------------------------- #
# Agent execution loop                                                         #
# --------------------------------------------------------------------------- #

def run_agent(
    instance: TaskInstance,
    group: List[Skill],
    eval_dir: str = "skills",
    model: str = "claude-haiku-4-5-20251001",
    max_completion_tokens: int = 4096,
    max_turns: int = 10,
) -> Dict[str, Any]:
    """
    Run the agent on a task with on-demand skill invocation.

    Returns:
        trace           – full message list (system omitted, user/assistant turns)
        final_answer    – extracted answer string
        agent_trace     – concatenated assistant text (for inventor)
        skill_usage     – Counter {skill_id: n_invocations}
        input_tokens    – total API usage
        output_tokens   – total API usage
    """
    valid_ids = {s.id for s in group}
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": _build_system(group)},
        {"role": "user", "content": instance.prompt},
    ]

    skill_usage: Counter = Counter()
    assistant_texts: List[str] = []
    input_tokens = 0
    output_tokens = 0
    final_answer = ""

    for _ in range(max_turns):
        response = _client().chat.completions.create(
            model=model,
            max_completion_tokens=max_completion_tokens,
            messages=messages,
        )
        text = response.choices[0].message.content or ""
        if response.usage:
            input_tokens += response.usage.prompt_tokens
            output_tokens += response.usage.completion_tokens

        messages.append({"role": "assistant", "content": text})
        assistant_texts.append(text)

        # Check for final answer first
        if "FINAL ANSWER:" in text.upper():
            final_answer = _extract_final_answer(text)
            break

        # Check for skill invocation
        match = _INVOKE_RE.search(text)
        if match:
            skill_id = match.group(1).lower()
            if skill_id in valid_ids:
                content = _load_skill_content(skill_id, eval_dir)
                if content:
                    skill_usage[skill_id] += 1
                    messages.append({
                        "role": "user",
                        "content": f"[Skill: {skill_id}]\n\n{content}\n\nNow continue solving the task.",
                    })
                    continue
                else:
                    messages.append({
                        "role": "user",
                        "content": f"Skill '{skill_id}' is not available. Please continue and end with FINAL ANSWER: <answer>",
                    })
                    continue

        # No invocation and no final answer — nudge once toward a final answer
        messages.append({
            "role": "user",
            "content": "Please end your response with  FINAL ANSWER: <your answer>",
        })

    return {
        "trace": messages,
        "final_answer": final_answer or assistant_texts[-1] if assistant_texts else "",
        "agent_trace": "\n\n---\n\n".join(assistant_texts),
        "skill_usage": skill_usage,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def _extract_final_answer(text: str) -> str:
    lines = text.strip().splitlines()
    for line in reversed(lines):
        if line.upper().startswith("FINAL ANSWER:"):
            return line.split(":", 1)[1].strip()
    return text.strip()
