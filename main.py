"""
Entry point for the Open-ended Semantic Skill Evolution framework.

Storage layout:
  skill_repo/   ← permanent repository: all skills + registry.json (V, N, logs)
  skills/       ← ephemeral eval workspace: only the sampled group (auto-managed)

Usage:
  python main.py                          # run 20 iterations with QA task
  python main.py --task minihack          # use MiniHack (requires BALROG Docker)
  python main.py --iterations 50
  python main.py --group-size 4
  python main.py --repo skill_repo        # custom repo directory
  python main.py --task minihack --docker-image balrog
"""

import argparse
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

from skillevo import SkillPool, EvolutionEngine, EvolutionConfig, SimpleQATask
from skillevo.tasks import MiniHackTask, BabaIsYouTask, TaskInstance


def main():
    parser = argparse.ArgumentParser(description="Semantic Skill Evolution")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--repo", default="skill_repo",
                        help="Permanent skill repository directory")
    parser.add_argument("--eval-dir", default="skills",
                        help="Ephemeral eval workspace (auto-managed)")
    parser.add_argument("--tau", type=float, default=2.0, help="Initial temperature")
    parser.add_argument("--alpha", type=float, default=0.1, help="EMA learning rate")
    parser.add_argument("--invent-every", type=int, default=5)
    parser.add_argument("--n-new", type=int, default=3, help="New skills per invention")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model", default="gpt-4o-mini",
        help="Fallback model for agent, credit, and invention",
    )
    parser.add_argument("--agent-model", default="", help="Model for agent episodes (overrides --model)")
    parser.add_argument("--invent-model", default="", help="Model for inventor + credit (overrides --model)")
    parser.add_argument(
        "--task", default="qa", choices=["qa", "minihack", "babaisai"],
        help="Task environment: 'qa', 'babaisai' (default local), or 'minihack' (requires BALROG Docker)",
    )
    parser.add_argument(
        "--docker-image", default="balrog",
        help="Docker image name for MiniHack episodes (default: balrog)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=30,
        help="Max steps per episode",
    )
    parser.add_argument(
        "--n-train", type=int, default=10,
        help="Number of BabaIsYou training tasks (rest are test set)",
    )
    parser.add_argument(
        "--n-eval", type=int, default=10,
        help="Episodes per iteration (run in parallel)",
    )
    parser.add_argument("--agent-api-key", default="", help="API key for agent episodes (default: ANTHROPIC_API_KEY env var)")
    parser.add_argument("--agent-base-url", default="", help="Base URL for agent API (default: Anthropic)")
    parser.add_argument("--invent-api-key", default="", help="API key for inventor+credit (default: ANTHROPIC_API_KEY env var)")
    parser.add_argument("--invent-base-url", default="", help="Base URL for inventor+credit API (default: Anthropic)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training; load saved pool and run test eval only")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY") \
            and not args.agent_api_key and not args.invent_api_key:
        print("Error: set OPENAI_API_KEY (or ANTHROPIC_API_KEY) environment variable.", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("  Open-ended Semantic Skill Evolution")
    print("=" * 60)

    # --- Task ---------------------------------------------------------- #
    if args.task == "babaisai":
        task = BabaIsYouTask(
            split="train",
            n_train=args.n_train,
            seed=args.seed,
            max_steps=args.max_steps,
            rng=__import__("numpy").random.default_rng(args.seed),
        )
        print(f"Task: BabaIsYou  (train={len(task.tasks)} / test={53 - len(task.tasks)}, max_steps={args.max_steps})")
    elif args.task == "minihack":
        task = MiniHackTask(
            docker_image=args.docker_image,
            max_steps=args.max_steps,
            rng=__import__("numpy").random.default_rng(args.seed),
        )
        print(f"Task: MiniHack  (docker={args.docker_image}, max_steps={args.max_steps})")
    else:
        task = SimpleQATask(seed=args.seed)

    # --- Skill pool ---------------------------------------------------- #
    pool = SkillPool(repo_dir=args.repo, eval_dir=args.eval_dir)
    print(f"\n{pool.summary()}\n")

    if len(pool) < args.group_size:
        print(
            f"Warning: pool has {len(pool)} skills but group_size={args.group_size}. "
            "Group size will be capped at pool size.",
        )

    # --- Config -------------------------------------------------------- #
    cfg = EvolutionConfig(
        group_size=args.group_size,
        tau_init=args.tau,
        alpha=args.alpha,
        n_eval=args.n_eval,
        invent_every=args.invent_every,
        n_new_skills=args.n_new,
        seed=args.seed,
        agent_model=args.agent_model or args.model,
        credit_model=args.invent_model or args.model,
        invent_model=args.invent_model or args.model,
        agent_api_key=args.agent_api_key,
        agent_base_url=args.agent_base_url,
        invent_api_key=args.invent_api_key,
        invent_base_url=args.invent_base_url,
    )

    # --- Eval-only mode (skip training, load saved pool) ------------------- #
    if args.eval_only:
        if args.task != "babaisai":
            print("--eval-only is only supported for --task babaisai", file=sys.stderr)
            sys.exit(1)
        from skillevo.evolution import _make_client
        _run_test_eval(pool, cfg, args,
                       agent_client=_make_client(cfg.agent_api_key, cfg.agent_base_url))
        return

    # EvolutionEngine creates the two clients and injects the agent client into
    # the task — no separate pre-warm needed.

    # --- Run ----------------------------------------------------------- #
    engine = EvolutionEngine(task=task, pool=pool, config=cfg)
    print(f"Running {args.iterations} iterations...\n")
    engine.run(n_iterations=args.iterations)

    print("\n" + "=" * 60)
    print("Evolution complete.  Final skill pool:\n")
    print(pool.summary())
    print(f"\nLogs saved to: evolution_logs/history.jsonl")
    print(f"Pool state saved to: {args.repo}/registry.json")

    # --- Test evaluation (BabaIsYou only) --------------------------------- #
    if args.task == "babaisai":
        _run_test_eval(pool, cfg, args, agent_client=engine._agent_client)


def _run_test_eval(pool, cfg, args, agent_client=None):
    """Evaluate the best skill group on all test tasks."""
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor, as_completed

    test_task = BabaIsYouTask(
        split="test",
        n_train=args.n_train,
        seed=args.seed,
        max_steps=args.max_steps,
    )
    if agent_client is not None:
        test_task._client = agent_client
    if not test_task.tasks:
        print("No test tasks available.")
        return

    # Use best group if recorded, otherwise top-V skills
    best = pool.best_group
    if best and best.get("skill_ids"):
        group = [s for s in pool.skills if s.id in best["skill_ids"]]
        print(f"\n{'='*60}")
        print(f"Test evaluation — best group from iter {best.get('iteration', '?')}")
    else:
        group = sorted(pool.skills, key=lambda s: -s.V)[: cfg.group_size]
        print(f"\n{'='*60}")
        print("Test evaluation — top-V skill group")

    print(f"Skills: {[s.name for s in group]}")
    print(f"Running on {len(test_task.tasks)} test tasks...\n")

    pool.stage_group(group)
    eval_dir = str(pool.eval_dir)

    def _run(task_name):
        inst = TaskInstance(
            id=f"test_{task_name.replace('/', '_')}",
            prompt=task_name,
            metadata={"task": task_name},
        )
        _, reward = test_task.run_episode(inst, group, eval_dir, cfg.agent_model)
        return task_name, reward

    results = {}
    with ThreadPoolExecutor(max_workers=len(test_task.tasks)) as executor:
        futures = {executor.submit(_run, t): t for t in test_task.tasks}
        for future in as_completed(futures):
            try:
                task_name, reward = future.result()
                results[task_name] = reward
                print(f"  {'✓' if reward > 0 else '✗'}  {task_name:<50}  reward={reward:.2f}")
            except Exception as e:
                task_name = futures[future]
                results[task_name] = 0.0
                print(f"  ✗  {task_name:<50}  reward=0.00  (error: {e})")

    pool.clear_eval()
    mean = sum(results.values()) / len(results)
    solved = sum(1 for r in results.values() if r > 0)
    print(f"\nTest results: {solved}/{len(results)} solved  |  mean reward = {mean:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
