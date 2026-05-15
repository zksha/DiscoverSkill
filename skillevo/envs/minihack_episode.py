#!/usr/bin/env python3
"""
MiniHack episode runner — executes inside the BALROG Docker container.

Per-step agent flow:
  1. Agent sees current observation + skill menu + valid actions.
  2. Agent may write  INVOKE_SKILL: <skill_id>  to load detailed guidance.
  3. After reviewing skills, agent ends with  ACTION: <action>
  4. Repeat until done or max_steps reached.

Output: JSON on stdout with keys:
  reward, skill_usage, agent_trace, messages, steps, input_tokens, output_tokens
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# MiniHack / BALROG imports (only available inside the Docker container)
# ---------------------------------------------------------------------------
try:
    import gym
    from balrog.environments.nle import AutoMore, NLELanguageWrapper
except ImportError as exc:
    sys.exit(f"MiniHack/BALROG not found — must run inside BALROG container: {exc}")

import anthropic

# ---------------------------------------------------------------------------
# Action vocabulary
# ---------------------------------------------------------------------------

_ALL_ACTIONS: dict[str, str] = {
    "north": "move north",
    "south": "move south",
    "east": "move east",
    "west": "move west",
    "northeast": "move northeast",
    "southeast": "move southeast",
    "southwest": "move southwest",
    "northwest": "move northwest",
    "far north": "move far north",
    "far south": "move far south",
    "far east": "move far east",
    "far west": "move far west",
    "up": "go up the stairs",
    "down": "go down the stairs",
    "wait": "rest one move",
    "search": "search for hidden doors",
    "kick": "kick door or enemy",
    "open": "open a door",
    "close": "close a door",
    "pickup": "pick up items here",
    "eat": "eat something",
    "quaff": "drink something",
    "pray": "pray to the gods",
}

_INVOKE_RE = re.compile(r"INVOKE_SKILL\s*:\s*([a-z0-9_]+)", re.IGNORECASE)
_ACTION_RE = re.compile(r"ACTION\s*:\s*(.+)", re.IGNORECASE)


def get_available_actions(env) -> dict[str, str]:
    """Return {action_key: description} for actions the env supports."""
    try:
        available = {}
        for action in env.actions:
            key = NLELanguageWrapper.all_nle_action_map[action][0]
            if key in _ALL_ACTIONS:
                available[key] = _ALL_ACTIONS[key]
        return available or _ALL_ACTIONS
    except Exception:
        return _ALL_ACTIONS


def get_goal(task: str) -> str:
    t = task.lower()
    if "corridor" in t:
        return "Explore the level and reach the stairs down."
    if "quest" in t:
        return "Explore, fight monsters, and reach the stairs down."
    if "boxoban" in t:
        return "Push all boulders onto fountains (Sokoban-style push mechanics)."
    if "mazewalk" in t:
        return "Navigate the maze and reach the stairs down."
    return "Get as far as possible in the game."


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_system(skill_menu: list[dict], available_actions: dict[str, str], goal: str) -> str:
    action_block = "\n".join(f"  {k}: {v}" for k, v in available_actions.items())
    skill_block = "\n".join(f"  [{s['id']}] {s['description']}" for s in skill_menu)
    return f"""\
You are playing MiniHack. Goal: {goal}

## Skills (optional strategic guidance)
{skill_block}

To load a skill's full instructions, write on its own line:
  INVOKE_SKILL: <skill_id>

You will receive the skill content immediately and can reason with it.

## Valid game actions
{action_block}

## Rules
1. End every response with exactly:  ACTION: <action_name>
   where <action_name> exactly matches one of the valid actions above.
2. You may invoke any skill before choosing your action.
3. Think before acting — do not repeat an action that just failed.
"""


def load_skill(skill_id: str, skills_dir: str) -> str | None:
    p = Path(skills_dir) / f"{skill_id}.md"
    return p.read_text(encoding="utf-8") if p.exists() else None


# ---------------------------------------------------------------------------
# Per-step agent decision
# ---------------------------------------------------------------------------

def agent_step(
    client: anthropic.Anthropic,
    system: str,
    obs_message: str,
    skill_menu: list[dict],
    skills_dir: str,
    model: str,
    max_tokens: int = 1024,
    max_skill_calls: int = 3,
) -> tuple[str, list[dict], Counter]:
    """
    One agent decision turn. May invoke skills before outputting an action.
    Returns (action, messages_this_turn, skill_usage_counter).
    """
    valid_skill_ids = {s["id"] for s in skill_menu}
    usage: Counter = Counter()
    messages: list[dict] = [{"role": "user", "content": obs_message}]

    for _ in range(max_skill_calls + 2):   # +2: initial call + fallback nudge
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )
        text = resp.content[0].text
        messages.append({"role": "assistant", "content": text})

        # Skill invocation takes priority
        inv_m = _INVOKE_RE.search(text)
        if inv_m:
            sid = inv_m.group(1).lower()
            if sid in valid_skill_ids:
                usage[sid] += 1
                content = load_skill(sid, skills_dir)
                if content:
                    messages.append({
                        "role": "user",
                        "content": (
                            f"[Skill: {sid}]\n\n{content}\n\n"
                            "Now choose your action and end with  ACTION: <action_name>"
                        ),
                    })
                    continue

        # Extract ACTION:
        act_m = _ACTION_RE.search(text)
        if act_m:
            return act_m.group(1).strip().lower(), messages, usage

        # Nudge the agent to produce an action
        messages.append({
            "role": "user",
            "content": "Please decide on an action and end your response with  ACTION: <action_name>",
        })

    return "wait", messages, usage  # fallback


# ---------------------------------------------------------------------------
# Episode loop
# ---------------------------------------------------------------------------

def run_episode(
    task_name: str,
    skill_menu: list[dict],
    skills_dir: str,
    model: str,
    max_steps: int,
) -> dict:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    env = gym.make(
        task_name,
        observation_keys=[
            "glyphs", "blstats", "tty_chars",
            "inv_letters", "inv_strs", "tty_cursor", "tty_colors",
        ],
        character="@",
        max_episode_steps=max_steps * 3,
        penalty_step=-0.01,
        autopickup=False,
    )
    env = AutoMore(env)
    env = NLELanguageWrapper(env)

    obs_raw = env.reset()
    available_actions = get_available_actions(env)
    goal = get_goal(task_name)
    system = build_system(skill_menu, available_actions, goal)

    total_reward = 0.0
    skill_usage: Counter = Counter()
    all_messages: list[dict] = []
    trace_parts: list[str] = []
    input_tok = output_tok = 0
    recent_actions: list[str] = []

    obs_text = obs_raw.get("text", obs_raw) if isinstance(obs_raw, dict) else {}

    for step in range(max_steps):
        long_term = obs_text.get("long_term_context", "")
        short_term = obs_text.get("short_term_context", "")
        history_line = (
            f"Recent actions: {', '.join(recent_actions[-5:])}\n\n"
            if recent_actions else ""
        )
        obs_msg = (
            f"Step {step + 1}/{max_steps}\n"
            f"{history_line}"
            f"Inventory/Stats:\n{short_term}\n\n"
            f"Observation:\n{long_term}"
        )

        action, step_msgs, step_usage = agent_step(
            client, system, obs_msg, skill_menu, skills_dir, model,
        )

        for k, v in step_usage.items():
            skill_usage[k] += v

        # Collect assistant texts for trace
        assistant_texts = [m["content"] for m in step_msgs if m["role"] == "assistant"]
        trace_parts.append(
            f"[Step {step + 1}]  ACTION: {action}\n"
            + "\n---\n".join(assistant_texts)
        )
        all_messages.extend(step_msgs)

        # Validate action
        if action not in available_actions:
            action = "wait"
        recent_actions.append(action)

        step_result = env.step(action)
        # gym v0.21 returns 4 values; newer returns 5 (terminated, truncated)
        if len(step_result) == 5:
            obs_raw, reward, terminated, truncated, _ = step_result
            done = terminated or truncated
        else:
            obs_raw, reward, done, _ = step_result

        total_reward += reward
        obs_text = obs_raw.get("text", obs_raw) if isinstance(obs_raw, dict) else {}

        if done:
            step += 1  # count the final step
            break

    env.close()

    # Normalise to [0, 1]: map [-1, +1] episode return linearly
    norm_reward = float(max(0.0, min(1.0, (total_reward + 1.0) / 2.0)))

    return {
        "reward": norm_reward,
        "raw_reward": total_reward,
        "skill_usage": dict(skill_usage),
        "agent_trace": "\n\n---\n\n".join(trace_parts),
        "messages": all_messages,
        "steps": step + 1,
        "input_tokens": input_tok,
        "output_tokens": output_tok,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run one MiniHack episode with skillevo skills.")
    parser.add_argument("--task", required=True, help="e.g. MiniHack-Corridor-R3-v0")
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--skill-menu", required=True,
                        help='JSON list: [{"id":..,"name":..,"description":..}, ...]')
    parser.add_argument("--skills-dir", default="/skills",
                        help="Directory containing skill .md files")
    args = parser.parse_args()

    skill_menu = json.loads(args.skill_menu)
    result = run_episode(
        task_name=args.task,
        skill_menu=skill_menu,
        skills_dir=args.skills_dir,
        model=args.model,
        max_steps=args.max_steps,
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
