"""Abstract task interface — swap in any environment here."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class TaskInstance:
    id: str
    prompt: str                       # what the agent sees
    metadata: Dict[str, Any]          # ground truth etc. (opaque to agent)


@dataclass
class ExecutionResult:
    task_id: str
    reward: float                     # scalar ∈ [0, 1]
    trace: List[Dict[str, Any]]       # message history for credit assignment
    info: Dict[str, Any]              # any additional info


class BaseTask(ABC):
    """Override this class to plug in a new task environment."""

    @abstractmethod
    def sample(self) -> TaskInstance:
        """Return a random task instance."""

    @abstractmethod
    def evaluate(self, instance: TaskInstance, response: str) -> float:
        """Return reward ∈ [0, 1] given the agent's final response."""

    def env_description(self) -> str:
        """Short description of the environment for the skill inventor."""
        return ""

    def run_episode(
        self,
        instance: "TaskInstance",
        group: List[Any],
        eval_dir: str,
        model: str,
    ) -> Tuple[Dict[str, Any], float]:
        """
        Run a full episode and return (result_dict, reward).

        Default implementation: Q&A-style via run_agent() + evaluate().
        Override for interactive environments (e.g. MiniHack).
        """
        from ..agent import run_agent
        result = run_agent(instance, group, eval_dir=eval_dir, model=model)
        reward = self.evaluate(instance, result["final_answer"])
        return result, reward
