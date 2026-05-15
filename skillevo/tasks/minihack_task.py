"""
MiniHack task — episodes run inside the BALROG Docker container.

Each call to run_episode():
  1. Mounts the staged skills/ directory into the container.
  2. Runs minihack_episode.py inside Docker via subprocess.
  3. Parses the JSON result and returns it in the standard format.

Docker image must be built from /home/leah/alma/envs_docker/BALROG/Dockerfile
and tagged to match the `docker_image` config option (default: "balrog").
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import BaseTask, TaskInstance
from ..skill import Skill

logger = logging.getLogger(__name__)

# Default curated task list (covers corridor, maze, quest, boxoban categories)
MINIHACK_TASKS: List[str] = [
    "MiniHack-Corridor-R3-v0",
    "MiniHack-Corridor-R5-v0",
    "MiniHack-MazeWalk-9x9-v0",
    "MiniHack-MazeWalk-15x15-v0",
    "MiniHack-Quest-Easy-v0",
    "MiniHack-Quest-Medium-v0",
    "MiniHack-Boxoban-Medium-v0",
]

# Absolute path to the episode runner script (mounted into Docker)
_EPISODE_SCRIPT = Path(__file__).resolve().parent.parent / "envs" / "minihack_episode.py"


class MiniHackTask(BaseTask):
    """
    Evaluates skill groups on MiniHack via the BALROG Docker container.

    run_episode() launches Docker, runs one full episode with the staged
    skills available for INVOKE_SKILL, and returns the reward + trace.
    """

    def __init__(
        self,
        tasks: List[str] = MINIHACK_TASKS,
        docker_image: str = "balrog",
        max_steps: int = 100,
        docker_timeout: int = 600,    # seconds before giving up on Docker call
        rng: Optional[np.random.Generator] = None,
    ):
        self.tasks = list(tasks)
        self.docker_image = docker_image
        self.max_steps = max_steps
        self.docker_timeout = docker_timeout
        self._rng = rng or np.random.default_rng(42)

    # ------------------------------------------------------------------ #
    # BaseTask interface                                                    #
    # ------------------------------------------------------------------ #

    def sample(self) -> TaskInstance:
        task = str(self._rng.choice(self.tasks))
        return TaskInstance(
            id=f"minihack_{uuid.uuid4().hex[:8]}",
            prompt=task,                       # task gym ID is the "prompt"
            metadata={"task": task},
        )

    def evaluate(self, instance: TaskInstance, response: str) -> float:
        """Extract reward embedded in the JSON response string."""
        try:
            return float(json.loads(response)["reward"])
        except (json.JSONDecodeError, TypeError, KeyError):
            return 0.0

    # ------------------------------------------------------------------ #
    # Interactive episode (overrides BaseTask.run_episode)                 #
    # ------------------------------------------------------------------ #

    def run_episode(
        self,
        instance: TaskInstance,
        group: List[Skill],
        eval_dir: str,
        model: str,
    ) -> Tuple[Dict[str, Any], float]:
        """
        Run one MiniHack episode inside Docker.

        Docker mounts:
          <eval_dir>          → /skills   (read-only, staged skill .md files)
          minihack_episode.py → /episode.py (read-only, the runner script)

        Returns (result_dict, reward) matching the format of run_agent().
        """
        task_name = instance.metadata["task"]

        skill_menu = [
            {"id": s.id, "name": s.name, "description": s.description}
            for s in group
        ]

        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        skills_abs = str(Path(eval_dir).resolve())
        script_abs = str(_EPISODE_SCRIPT)

        cmd = [
            "docker", "run", "--rm",
            "-e", f"ANTHROPIC_API_KEY={api_key}",
            "-v", f"{skills_abs}:/skills:ro",
            "-v", f"{script_abs}:/episode.py:ro",
            self.docker_image,
            "python", "/episode.py",
            "--task", task_name,
            "--model", model,
            "--max-steps", str(self.max_steps),
            "--skill-menu", json.dumps(skill_menu),
            "--skills-dir", "/skills",
        ]

        logger.debug("Launching Docker episode: %s", task_name)
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.docker_timeout,
            )
        except subprocess.TimeoutExpired:
            logger.error("Docker episode timed out after %ds (%s)", self.docker_timeout, task_name)
            return _empty_result(instance), 0.0
        except FileNotFoundError:
            logger.error("'docker' not found — is Docker installed and on PATH?")
            return _empty_result(instance), 0.0

        if proc.returncode != 0:
            logger.error(
                "Docker episode exited %d for %s:\nSTDERR:\n%s",
                proc.returncode, task_name, proc.stderr[-3000:],
            )
            return _empty_result(instance), 0.0

        # Parse JSON from stdout
        stdout = proc.stdout.strip()
        try:
            raw = json.loads(stdout)
        except json.JSONDecodeError:
            logger.error("Could not parse episode JSON:\n%s", stdout[:500])
            return _empty_result(instance), 0.0

        reward = float(raw.get("reward", 0.0))
        logger.info(
            "Episode done: task=%s  reward=%.3f  steps=%d  skills=%s",
            task_name, reward, raw.get("steps", 0), raw.get("skill_usage", {}),
        )

        result = {
            "trace": _to_trace(raw.get("messages", [])),
            "final_answer": json.dumps({"reward": reward}),
            "agent_trace": raw.get("agent_trace", ""),
            "skill_usage": Counter(raw.get("skill_usage", {})),
            "input_tokens": raw.get("input_tokens", 0),
            "output_tokens": raw.get("output_tokens", 0),
        }
        return result, reward


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_result(instance: TaskInstance) -> Dict[str, Any]:
    return {
        "trace": [{"role": "user", "content": instance.prompt}],
        "final_answer": json.dumps({"reward": 0.0}),
        "agent_trace": "",
        "skill_usage": Counter(),
        "input_tokens": 0,
        "output_tokens": 0,
    }


def _to_trace(messages: List[Dict]) -> List[Dict[str, str]]:
    """Normalise episode messages to the credit-assignment trace format."""
    return [
        {"role": m.get("role", "user"), "content": str(m.get("content", ""))}
        for m in messages
    ]
