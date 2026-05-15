"""
Main Evolution Loop — orchestrates the full Open-ended Semantic Skill Evolution.

One iteration:
  1. Sample skill group G (temperature-controlled softmax + freshness bonus)
  2. Run agent on a task instance with G
  3. Score → reward R_G
  4. Semantic credit assignment: C(s_i) for each s_i ∈ G
  5. Group-relative advantage: Â(s_i) = C(s_i) · (R_G − baseline(s_i))
  6. EMA value update: V(s_i) ← (1−α)·V(s_i) + α·Â(s_i)
  7. Every `invent_every` iterations: LLM invents `n_new` skills
  8. Adaptive temperature via sampling entropy
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import openai
import numpy as np

from .skill import Skill
from .skill_pool import SkillPool
from .sampler import sample_group, adapt_temperature
from .credit import assign_credit
from .inventor import invent_skills
from .tasks.base import BaseTask, TaskInstance

logger = logging.getLogger(__name__)


@dataclass
class EvolutionConfig:
    # Sampling
    group_size: int = 6
    tau_init: float = 2.0
    tau_min: float = 0.1
    tau_max: float = 5.0
    tau_delta: float = 0.15
    c: float = 1.0                   # UCB exploration coefficient

    # Value update
    alpha: float = 0.1              # EMA learning rate: V ← (1−α)V + α·(C·R)

    # Evaluation
    n_eval: int = 10                 # episodes per iteration (run in parallel)

    # Invention
    invent_every: int = 2           # invent new skills every N iterations
    n_new_skills: int = 5

    # Pool limits
    max_pool_size: int = 50

    # Models
    agent_model: str = "claude-haiku-4-5-20251001"
    credit_model: str = "claude-haiku-4-5-20251001"
    invent_model: str = "claude-haiku-4-5-20251001"

    # API credentials — agent (task episodes) vs invent (inventor + credit)
    # Empty string → fall back to ANTHROPIC_API_KEY env var and default base URL
    agent_api_key: str = ""
    agent_base_url: str = ""
    invent_api_key: str = ""
    invent_base_url: str = ""

    # Logging
    log_dir: str = "evolution_logs"
    seed: int = 42


@dataclass
class IterationRecord:
    iteration: int
    task_id: str
    task_prompt: str
    skill_ids: List[str]
    skill_names: List[str]
    reward: float
    episode_rewards: List[float]      # per-episode rewards (len = n_eval)
    contributions: Dict[str, float]   # C(s_i) ∈ [0, 1]
    value_updates: Dict[str, float]   # ΔV per skill
    agent_trace: str                  # full tagged reasoning from agent
    tau: float
    entropy: float
    new_skills_invented: List[str]
    elapsed_seconds: float


def _make_client(api_key: str = "", base_url: str = "") -> openai.OpenAI:
    import os
    key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY", "")
    kwargs: Dict[str, Any] = {"api_key": key}
    if base_url:
        kwargs["base_url"] = base_url
    return openai.OpenAI(**kwargs)


class EvolutionEngine:
    def __init__(
        self,
        task: BaseTask,
        pool: SkillPool,
        config: EvolutionConfig | None = None,
    ):
        self.task = task
        self.pool = pool
        self.cfg = config or EvolutionConfig()
        self.tau = self.cfg.tau_init
        self._rng = np.random.default_rng(self.cfg.seed)
        self._history: List[Dict[str, Any]] = []

        # Two separate API clients: one for agent episodes, one for inventor+credit
        self._agent_client = _make_client(self.cfg.agent_api_key, self.cfg.agent_base_url)
        self._invent_client = _make_client(self.cfg.invent_api_key, self.cfg.invent_base_url)

        # Inject agent client into the task (BabaIsYouTask stores it as _client)
        if hasattr(task, "_client"):
            task._client = self._agent_client

        # Best group tracker: highest single-iteration reward seen so far
        self._best: Dict[str, Any] = {}   # {skill_ids, reward, iteration, task_prompt}

        log_path = Path(self.cfg.log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        self._log_path = log_path / "history.jsonl"

    # ------------------------------------------------------------------ #
    # Main loop                                                            #
    # ------------------------------------------------------------------ #

    def run(self, n_iterations: int) -> None:
        logger.info(
            "Starting evolution: %d iterations, pool_size=%d",
            n_iterations, len(self.pool),
        )
        for i in range(n_iterations):
            record = self._step(self.pool.generation + 1)
            self.pool.generation += 1
            self._history.append(self._record_to_history(record))
            self._log(record)
            self._update_best(record)
            self.pool.save()
            self._print_step(record)

        self._report_best()

    def _step(self, iteration: int) -> IterationRecord:
        t0 = time.perf_counter()
        skills = self.pool.skills

        if len(skills) < 1:
            raise RuntimeError("Skill pool is empty — add seed skills first.")

        # ---- 1. Sample group ----------------------------------------- #
        group, probs, H = sample_group(
            skills, k=self.cfg.group_size, tau=self.tau,
            c=self.cfg.c, rng=self._rng,
        )
        logger.info("── iter %d  group=[%s]  τ=%.2f",
                    iteration, ", ".join(s.name for s in group), self.tau)

        # ---- 2. Stage eval workspace (skills/ = sampled group only) -- #
        self.pool.stage_group(group)

        # ---- 3. Sample n_eval task instances ------------------------- #
        instances = [self.task.sample() for _ in range(self.cfg.n_eval)]
        logger.info("   Running %d episodes in parallel...", len(instances))

        # ---- 4. Run episodes in parallel ----------------------------- #
        eval_dir = str(self.pool.eval_dir)
        model = self.cfg.agent_model

        def _run(inst):
            return self.task.run_episode(inst, group, eval_dir, model)

        episode_results: List[Tuple[Dict, float]] = []
        with ThreadPoolExecutor(max_workers=self.cfg.n_eval) as executor:
            futures = {executor.submit(_run, inst): inst for inst in instances}
            for future in as_completed(futures):
                try:
                    episode_results.append(future.result())
                except Exception as e:
                    logger.error("Episode failed: %s", e)
                    episode_results.append(({
                        "trace": [], "agent_trace": "",
                        "skill_usage": Counter(),
                    }, 0.0))

        # ---- 5. Aggregate across episodes ---------------------------- #
        rewards = [r for _, r in episode_results]
        reward = sum(rewards) / len(rewards)
        solved = sum(1 for r in rewards if r > 0)
        logger.info("   Episodes done: %d/%d solved  mean_reward=%.2f",
                    solved, len(rewards), reward)

        combined_usage: Counter = Counter()
        for res, _ in episode_results:
            combined_usage.update(res.get("skill_usage", Counter()))

        # Sample 3 complete episode traces for the inventor.
        # Random sampling gives the inventor diverse reasoning examples
        # rather than always seeing the same episodes.
        _traced = [(res, r) for res, r in episode_results if res.get("agent_trace")]
        _n_sample = min(3, len(_traced))
        _sampled_idx = self._rng.choice(len(_traced), size=_n_sample, replace=False)
        combined_trace = "\n\n=== episode boundary ===\n\n".join(
            _traced[i][0]["agent_trace"] for i in _sampled_idx
        )
        combined_messages = []
        for res, _ in episode_results:
            combined_messages.extend(res.get("trace", []))

        instance = instances[0]   # representative for logging

        # Build per-episode feedback: task name + reward for all n_eval episodes
        all_task_prompts = "\n".join(
            f"  {inst.prompt}  →  reward={r:.2f}"
            for inst, (_, r) in zip(instances, episode_results)
        )

        # ---- 6. Credit assignment ------------------------------------ #
        logger.info("   Credit assignment (model=%s)...", self.cfg.credit_model)
        contributions = assign_credit(
            task_prompt=instance.prompt,
            group=group,
            trace=combined_messages,
            usage=combined_usage,
            reward=reward,
            model=self.cfg.credit_model,
            agent_trace_text=combined_trace or None,
            client=self._invent_client,
        )
        credit_str = "  ".join(
            f"{s.name}={contributions.get(s.id, 0.0):.2f}" for s in group
        )
        logger.info("   Credit: %s", credit_str)

        # ---- 7. Credit-allocated reward + V update ------------------- #
        # r_i = C(s_i) · R_G  — reward allocated to each skill by credit
        # V(s_i) ← (1−α)·V + α·r_i  — EMA of credit-weighted reward
        #
        # C=0 (skill not tagged) → r_i=0, skip V update.
        # No separate baseline needed: V converges to E[C_i · R_G] naturally.
        value_updates: Dict[str, float] = {}
        for s in group:
            C = contributions.get(s.id, 0.0)
            r_i = C * reward
            s.N += 1
            if C > 0:
                old_V = s.V
                s.V = (1 - self.cfg.alpha) * s.V + self.cfg.alpha * r_i
                value_updates[s.id] = s.V - old_V
            else:
                value_updates[s.id] = 0.0
            s.add_log({
                "iter": iteration,
                "task_id": instance.id,
                "reward": reward,
                "contribution": C,
                "allocated_reward": r_i,
                "V_after": s.V,
                "V_updated": C > 0,
            })

        # ---- 8. Clear eval workspace --------------------------------- #
        self.pool.clear_eval()

        # ---- 9. Adaptive temperature --------------------------------- #
        self.tau = adapt_temperature(
            self.tau, H,
            n_skills=len(skills),
            tau_min=self.cfg.tau_min,
            tau_max=self.cfg.tau_max,
            delta=self.cfg.tau_delta,
        )

        elapsed = time.perf_counter() - t0

        # ---- 10. Skill invention ------------------------------------- #
        # Must come after IterationRecord is built (inventor needs record).
        record = IterationRecord(
            iteration=iteration,
            task_id=instance.id,
            task_prompt=all_task_prompts,
            skill_ids=[s.id for s in group],
            skill_names=[s.name for s in group],
            reward=reward,
            episode_rewards=rewards,
            contributions=contributions,
            value_updates=value_updates,
            agent_trace=combined_trace,
            tau=self.tau,
            entropy=H,
            new_skills_invented=[],
            elapsed_seconds=elapsed,
        )

        if iteration % self.cfg.invent_every == 0:
            logger.info("   Inventing skills (model=%s)...", self.cfg.invent_model)
            record.new_skills_invented = self._invent(record)
            if record.new_skills_invented:
                logger.info("   Invented: %s", ", ".join(record.new_skills_invented))
            else:
                logger.info("   Invention produced no new skills.")

        return record

    # ------------------------------------------------------------------ #
    # Invention                                                            #
    # ------------------------------------------------------------------ #

    def _invent(self, record: IterationRecord) -> List[str]:
        if len(self.pool) >= self.cfg.max_pool_size:
            logger.info("Pool at max size (%d), skipping invention.", self.cfg.max_pool_size)
            return []

        slots = self.cfg.max_pool_size - len(self.pool)
        n_new = min(self.cfg.n_new_skills, slots)

        # Build per-skill detail from last eval for the inventor
        group_detail = []
        for sid, sname in zip(record.skill_ids, record.skill_names):
            skill = self.pool.get(sid)
            group_detail.append({
                "id": sid,
                "name": sname,
                "description": skill.description if skill else "",
                "contribution": record.contributions.get(sid, 0.0),
                "V": skill.V if skill else 0.0,
            })

        proposals = invent_skills(
            pool_summary=self.pool.summary(),
            last_group=group_detail,
            last_reward=record.reward,
            last_task=record.task_prompt,
            agent_trace=record.agent_trace,
            n_new=n_new,
            model=self.cfg.invent_model,
            client=self._invent_client,
            env_description=self.task.env_description(),
        )

        added = []
        for p in proposals:
            skill = self.pool.add_skill(
                name=p["name"],
                description=p["description"],
                content=p["content"],
            )
            added.append(skill.name)
            logger.info("Invented skill: %s", skill.name)

        return added

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _record_to_history(self, rec: IterationRecord) -> Dict[str, Any]:
        return {
            "iteration": rec.iteration,
            "task_prompt": rec.task_prompt,
            "skill_names": rec.skill_names,
            "reward": rec.reward,
        }

    def _log(self, rec: IterationRecord) -> None:
        id_to_name = dict(zip(rec.skill_ids, rec.skill_names))
        credit = {
            id_to_name.get(sid, sid): round(c, 4)
            for sid, c in rec.contributions.items()
        }
        with self._log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "iteration": rec.iteration,
                "group": [
                    {"id": sid, "name": name}
                    for sid, name in zip(rec.skill_ids, rec.skill_names)
                ],
                "episode_rewards": [round(r, 4) for r in rec.episode_rewards],
                "n_solved": sum(1 for r in rec.episode_rewards if r > 0),
                "avg_reward": round(rec.reward, 4),
                "credit": credit,
                "tau": round(rec.tau, 4),
                "entropy": round(rec.entropy, 4),
                "new_skills": rec.new_skills_invented,
                "elapsed_s": round(rec.elapsed_seconds, 2),
            }) + "\n")

    def _update_best(self, rec: IterationRecord) -> None:
        if not self._best or rec.reward > self._best["reward"]:
            self._best = {
                "iteration": rec.iteration,
                "reward": rec.reward,
                "skill_ids": rec.skill_ids,
                "skill_names": rec.skill_names,
                "task_prompt": rec.task_prompt,
            }
            self.pool.set_best_group(self._best)

    def _report_best(self) -> None:
        if not self._best:
            return
        print("\n" + "=" * 60)
        print("Best group found:")
        print(f"  iteration : {self._best['iteration']}")
        print(f"  reward    : {self._best['reward']:.3f}")
        print(f"  skills    : {self._best['skill_names']}")
        print(f"  task      : {self._best['task_prompt']}")
        print("=" * 60)

    def _print_step(self, rec: IterationRecord) -> None:
        skills_str = ", ".join(rec.skill_names)
        new_str = (
            f" | invented=[{', '.join(rec.new_skills_invented)}]"
            if rec.new_skills_invented else ""
        )
        print(
            f"[iter {rec.iteration:04d}] "
            f"reward={rec.reward:.2f} | "
            f"τ={rec.tau:.2f} H={rec.entropy:.2f} | "
            f"skills=[{skills_str}]{new_str} | "
            f"{rec.elapsed_seconds:.1f}s"
        )
