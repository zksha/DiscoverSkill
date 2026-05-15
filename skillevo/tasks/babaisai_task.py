"""
BabaIsYou task — runs episodes locally (no Docker needed).

Uses the BALROG BabaIsAI wrapper pattern but text-only (no image rendering),
which is thread-safe for parallel episode execution.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Suppress pygame display requirement before any baba import
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import openai
import baba
import gym
from baba import make as baba_make
from baba.world_object import name_mapping

from .base import BaseTask, TaskInstance
from ..skill import Skill

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Task list and train/test split
# ---------------------------------------------------------------------------

try:
    ALL_TASKS: List[str] = list(baba_make("env/*").keys())
except Exception:
    ALL_TASKS = [
        "env/goto_win", "env/goto_win-distr_obj", "env/goto_win-distr_rule",
        "env/make_win", "env/make_win-distr_obj", "env/make_win-distr_rule",
        "env/two_room-goto_win", "env/two_room-make_win",
        "env/you_win", "env/you_win-fixed_you",
    ]


def make_split(
    tasks: List[str] = ALL_TASKS,
    n_train: int = 10,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """
    Stratified train/test split by task category.

    Categories: goto_win, make_win, two_room, you_win (inferred from task name).
    Within each category tasks are shuffled, then proportional n_train slots
    are filled until the quota is met.  Returns (train_tasks, test_tasks).
    """
    from collections import defaultdict
    import random as _random

    def _category(t: str) -> str:
        if "two_room" in t:
            return "two_room"
        if "make_win" in t:
            return "make_win"
        if "goto_win" in t:
            return "goto_win"
        return "other"

    buckets: dict = defaultdict(list)
    for t in tasks:
        buckets[_category(t)].append(t)

    rng = _random.Random(seed)
    for v in buckets.values():
        rng.shuffle(v)

    train, test = [], []
    remaining = n_train
    cats = sorted(buckets.keys())

    for i, cat in enumerate(cats):
        cat_tasks = buckets[cat]
        # Proportional share, give remainder to last category
        share = round(len(cat_tasks) / len(tasks) * n_train)
        if i == len(cats) - 1:
            share = remaining          # use up whatever is left
        share = min(share, len(cat_tasks), remaining)
        train += cat_tasks[:share]
        test += cat_tasks[share:]
        remaining -= share

    # If we still need more train tasks (rounding), pull from test
    while remaining > 0 and test:
        train.append(test.pop(0))
        remaining -= 1

    return train, test


# Precomputed default split (10 train / 43 test)
TRAIN_TASKS, TEST_TASKS = make_split(ALL_TASKS, n_train=10, seed=42)

ACTIONS: List[str] = [a.name for a in baba.grid.BabaIsYouEnv.Actions]
# e.g. ["idle", "up", "right", "down", "left"]

ACTION_DESCS: Dict[str, str] = {
    "idle":  "wait for one step",
    "up":    "take one step up",
    "right": "take one step to the right",
    "down":  "take one step down",
    "left":  "take one step to the left",
}

_INVOKE_RE = re.compile(r"INVOKE_SKILL\s*:\s*([a-z0-9_]+)", re.IGNORECASE)
_ACTION_RE = re.compile(r"ACTION\s*:\s*(\w+)", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Lightweight text-only env wrapper (no pygame rendering)
# ---------------------------------------------------------------------------

class _BabaTextWrapper(gym.Wrapper):
    """Text-only BabaIsYou wrapper — thread-safe, no image rendering."""

    def __init__(self, env: gym.Env, add_ruleset: bool = True):
        super().__init__(env)
        self.add_ruleset = add_ruleset
        self._action_names = ACTIONS[:]

    # -- observation helpers ------------------------------------------------

    def _get_ruleset(self) -> str:
        rules = []
        for rule in self.env.grid._ruleset["_rule_"]:
            if "object" not in rule:
                continue
            name = rule["object"].removeprefix("f")
            prop = name_mapping[rule["property"]]
            rules.append(f"{name} is {prop}")
        return "\n".join(rules)

    def _find_objects(self, types) -> List[Tuple]:
        objs = []
        for j in range(self.env.height):
            for i in range(self.env.width):
                cell = self.env.grid.get(i, j)
                if cell and cell.type in types:
                    if cell.type == "rule_object":
                        name = f"rule `{cell.name}`"
                    elif cell.type == "rule_is":
                        name = f"rule `{name_mapping[cell.name]}`"
                    elif cell.type == "rule_property":
                        name = f"rule `{name_mapping[cell.property]}`"
                    else:
                        name = cell.type
                    objs.append(((i, j), name))
        return objs

    def _get_text_obs(self) -> Tuple[str, bool]:
        you = None
        for rule in self.env.grid._ruleset["_rule_"]:
            if "property" not in rule:
                continue
            if name_mapping[rule["property"]] == "you":
                you = rule["object"]

        if you is None:
            return "[...] IS YOU rule broken — no controllable object.", True

        my_objs = self._find_objects([you])
        if not my_objs:
            return "[...] IS YOU rule broken — no controllable object.", True
        my_pos = my_objs[0][0]

        others = self._find_objects(
            ["fball", "fwall", "fdoor", "fkey", "rule_object", "rule_is", "rule_property"]
        )
        lines = []
        for (ox, oy), name in others:
            dx, dy = ox - my_pos[0], oy - my_pos[1]
            parts = []
            if dx > 0:
                parts.append(f"{dx} step{'s' if dx > 1 else ''} to the right")
            elif dx < 0:
                parts.append(f"{-dx} step{'s' if -dx > 1 else ''} to the left")
            if dy > 0:
                parts.append(f"{dy} step{'s' if dy > 1 else ''} down")
            elif dy < 0:
                parts.append(f"{-dy} step{'s' if -dy > 1 else ''} up")
            if parts:
                lines.append(f"{name} {' and '.join(parts)}")
        return "\n".join(lines) if lines else "(nothing nearby)", False

    def _make_text(self) -> str:
        text_obs, broken = self._get_text_obs()
        parts = []
        if broken:
            parts.append(text_obs)
        if self.add_ruleset:
            parts.append(f"Active rules:\n{self._get_ruleset()}")
        parts.append(f"Objects on the map:\n{text_obs if not broken else '(reset)'}")
        return "\n\n".join(parts)

    # -- gym interface ------------------------------------------------------

    def reset(self, **kwargs) -> str:
        self.env.reset(**kwargs)
        return self._make_text()

    def step(self, action: str) -> Tuple[str, float, bool, dict]:
        action_int = self._action_names.index(action)
        _, reward, done, info = self.env.step(action_int)
        return self._make_text(), reward, done, info


# ---------------------------------------------------------------------------
# Per-step agent decision
# ---------------------------------------------------------------------------

def _load_skill(skill_id: str, eval_dir: str) -> Optional[str]:
    from pathlib import Path
    p = Path(eval_dir) / f"{skill_id}.md"
    return p.read_text(encoding="utf-8") if p.exists() else None


def _agent_step(
    client: openai.OpenAI,
    system: str,
    obs_text: str,
    step_history: List[Dict],
    skill_menu: List[Dict],
    eval_dir: str,
    model: str,
    max_completion_tokens: int = 4096,
    max_skill_calls: int = 2,
) -> Tuple[str, List[Dict], Counter]:
    """One agent decision. Returns (action, messages, skill_usage)."""
    valid_skill_ids = {s["id"] for s in skill_menu}
    usage: Counter = Counter()

    if step_history:
        history_block = "\n\n".join(
            f"[Step {i + 1}] ACTION: {h['action']}\nObservation:\n{h['obs']}"
            for i, h in enumerate(step_history)
        ) + "\n\n"
    else:
        history_block = ""

    messages: List[Dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"{history_block}[Current]\nObservation:\n{obs_text}"},
    ]

    for _ in range(max_skill_calls + 2):
        resp = client.chat.completions.create(
            model=model, max_completion_tokens=max_completion_tokens, messages=messages,
        )
        text = resp.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": text})
        resp_usage = getattr(resp, "usage", None)
        if resp_usage:
            logger.debug("  model raw: %s | tokens: prompt=%s completion=%s reasoning=%s",
                         text[:200].replace("\n", " "),
                         getattr(resp_usage, "prompt_tokens", "?"),
                         getattr(resp_usage, "completion_tokens", "?"),
                         getattr(getattr(resp_usage, "completion_tokens_details", None), "reasoning_tokens", "?"))

        inv_m = _INVOKE_RE.search(text)
        if inv_m:
            sid = inv_m.group(1).lower()
            if sid in valid_skill_ids:
                content = _load_skill(sid, eval_dir)
                if content:
                    usage[sid] += 1
                    messages.append({
                        "role": "user",
                        "content": (
                            f"[Skill: {sid}]\n\n{content}\n\n"
                            "Now choose your action and end with  ACTION: <action>"
                        ),
                    })
                    continue
                else:
                    messages.append({
                        "role": "user",
                        "content": f"Skill '{sid}' is not available. Please choose your action and end with  ACTION: <action>",
                    })
                    continue

        act_m = _ACTION_RE.search(text)
        if act_m:
            return act_m.group(1).strip().lower(), messages, usage

        messages.append({
            "role": "user",
            "content": "Please end your response with  ACTION: <action>  (one of: " + ", ".join(ACTIONS) + "). Only output the legal action without any explaination.",
        })

    return "idle", messages, usage  # fallback


# ---------------------------------------------------------------------------
# BabaIsYouTask
# ---------------------------------------------------------------------------



class BabaIsYouTask(BaseTask):
    """
    BabaIsYou puzzle task — 53 levels, runs locally in-process.
    Episodes are short (typically 5–30 steps), no Docker required.

    split="train"  → sample only from TRAIN_TASKS (10 tasks, used during evolution)
    split="test"   → sample only from TEST_TASKS  (43 tasks, used for generalization eval)
    split="all"    → sample from all 53 tasks
    """

    def __init__(
        self,
        split: str = "train",
        n_train: int = 10,
        seed: int = 42,
        max_steps: int = 50,
        rng: Optional[np.random.Generator] = None,
    ):
        train, test = make_split(ALL_TASKS, n_train=n_train, seed=seed)
        if split == "train":
            self.tasks = train
        elif split == "test":
            self.tasks = test
        else:
            self.tasks = ALL_TASKS
        self.split = split
        self.max_steps = max_steps
        self._rng = rng or np.random.default_rng(seed)
        self._client: Optional[openai.OpenAI] = None
        # Shuffled epoch queue for without-replacement sampling
        self._epoch_queue: List[str] = []

    def _get_client(self) -> openai.OpenAI:
        if self._client is None:
            key = os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY", "")
            self._client = openai.OpenAI(api_key=key)
        return self._client

    def _ensure_client(self) -> None:
        self._get_client()

    # -- BaseTask -----------------------------------------------------------

    def env_description(self) -> str:
        return """\
## Environment: Baba Is You

Baba Is You is a grid-based puzzle game where the rules of the world are written as \
text blocks on the grid itself. Rules follow the pattern "[NOUN] IS [PROPERTY]" \
(e.g. "BABA IS YOU", "FLAG IS WIN", "WALL IS STOP").

Key mechanics:
- The player controls the object satisfying the "[X] IS YOU" rule.
- The goal is usually to reach (or become) the object satisfying "[X] IS WIN".
- Text blocks are physical objects — pushing them rearranges or breaks rules.
- Creating new rules (e.g. "KEY IS WIN") or destroying blocking rules \
(e.g. pushing STOP off a wall rule) is often the solution.
- A level fails if no "[X] IS YOU" rule exists or the YOU-object is stuck.

Task categories in this pool:
- goto_win: navigate to the WIN object without changing rules.
- make_win: rearrange text to create a WIN rule that is currently absent.
- two_room: two separated rooms; crossing requires rule manipulation.
- you_win: the YOU or WIN rule must be changed to solve the level.

Available actions: idle, up, right, down, left (one grid step each).\
"""

    def sample(self) -> TaskInstance:
        # Without-replacement sampling: shuffle a fresh epoch when the queue runs out.
        # With n_eval == len(tasks), every call to _step evaluates each task exactly once.
        if not self._epoch_queue:
            idx = self._rng.permutation(len(self.tasks))
            self._epoch_queue = [self.tasks[i] for i in idx]
        task = self._epoch_queue.pop()
        return TaskInstance(
            id=f"baba_{uuid.uuid4().hex[:8]}",
            prompt=task,
            metadata={"task": task},
        )

    def evaluate(self, instance: TaskInstance, response: str) -> float:
        try:
            return float(json.loads(response)["reward"])
        except (json.JSONDecodeError, TypeError, KeyError):
            return 0.0

    # -- Interactive episode ------------------------------------------------

    def run_episode(
        self,
        instance: TaskInstance,
        group: List[Skill],
        eval_dir: str,
        model: str,
    ) -> Tuple[Dict[str, Any], float]:
        task_name = instance.metadata["task"]
        client = self._get_client()

        skill_menu = [
            {"id": s.id, "name": s.name, "description": s.description}
            for s in group
        ]

        skill_block = "\n".join(f"  [{s['id']}] {s['description']}" for s in skill_menu)
        action_desc = "\n".join(f"{a}: {d}" for a, d in ACTION_DESCS.items())
        system = f"""\
Baba Is You is a puzzle game where you can manipulate the rules of each level. \
The following are the possible actions you can take in the game, \
followed by a short description of each action:

{action_desc}.

Tips:
- Examine the level carefully, noting all objects and text blocks present.
- Identify the current rules, which are formed by text blocks in the format "[Subject] IS [Property]" (e.g. "BABA IS YOU").
- Consider how you can change or create new rules by moving text blocks around.
- Remember that you can only move objects or text that are not defined as "STOP" or similar immovable properties.
- Your goal is usually to reach an object defined as "WIN", but this can be changed.
- Think creatively about how changing rules can alter the properties and behaviors of objects in unexpected ways.
- If stuck, try breaking apart existing rules or forming completely new ones.
- Sometimes the solution involves making yourself a different object or changing what counts as the win condition.

## Skills (optional strategic guidance)
You have access to the following skills that provide deeper reasoning guidance:
{skill_block}

To load a skill's full instructions, write on its own line:
  INVOKE_SKILL: <skill_id>

You may invoke a skill before deciding your action. \
After any reasoning, end your response with exactly:
  ACTION: <action>
"""

        # Create environment (each episode gets its own instance)
        env = _BabaTextWrapper(baba_make(task_name))
        obs_text = env.reset()

        total_reward = 0.0
        skill_usage: Counter = Counter()
        all_messages: List[Dict] = []
        trace_parts: List[str] = []
        step_history: List[Dict] = []   # {"obs": str, "action": str}
        solved = False
        tag = f"[{task_name.split('/')[-1]:<40}]"

        logger.info("%s start", tag)
        try:
            for step in range(self.max_steps):
                action, step_msgs, step_usage = _agent_step(
                    client, system, obs_text, step_history[-15:],
                    skill_menu, eval_dir, model,
                )
                for k, v in step_usage.items():
                    skill_usage[k] += v

                skills_used = list(step_usage.keys())
                skill_line = f"Skills invoked: {', '.join(skills_used)}" if skills_used else "Skills invoked: none"
                assistant_texts = [m["content"] for m in step_msgs if m["role"] == "assistant"]
                trace_parts.append(
                    f"[Step {step + 1}]\n"
                    f"Observation:\n{obs_text}\n"
                    f"{skill_line}\n"
                    f"ACTION: {action}\n"
                    + "\n---\n".join(assistant_texts)
                )
                all_messages.extend(step_msgs)

                skill_note = f"  (invoked: {', '.join(skills_used)})" if skills_used else ""
                logger.info("%s step %2d/%d  →  %s%s",
                            tag, step + 1, self.max_steps, action, skill_note)

                if action not in ACTIONS:
                    action = "idle"
                step_history.append({"obs": obs_text, "action": action})

                obs_text, reward, done, _ = env.step(action)
                total_reward += reward

                if done:
                    solved = reward > 0
                    break
        finally:
            env.close()

        reward_out = 1.0 if solved else 0.0
        status = "✓ solved" if solved else "✗ failed"
        logger.info("%s %s  (%d steps, skills used: %s)",
                    tag, status, len(trace_parts),
                    ", ".join(f"{k}×{v}" for k, v in skill_usage.items()) or "none")

        result = {
            "trace": all_messages,
            "final_answer": json.dumps({"reward": reward_out}),
            "agent_trace": "\n\n---\n\n".join(trace_parts),
            "skill_usage": skill_usage,
            "input_tokens": 0,
            "output_tokens": 0,
        }
        return result, reward_out
