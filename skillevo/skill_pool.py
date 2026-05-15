"""
SkillPool — two-layer skill storage.

skill_repo/          ← permanent repository: ALL skills (content + state)
  registry.json      ← full state for every skill (V, N, logs, timestamps…)
  chain_of_thought.md
  invented_skill_001.md
  …

skills/              ← ephemeral eval workspace: ONLY the current sampled group
  chain_of_thought.md
  decompose_problem.md
  …  (cleared and rewritten before every agent eval)
"""

from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .skill import Skill


_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n?", re.DOTALL)


def _parse_md(path: Path) -> tuple[str, str, str]:
    """Return (name, description, body) from a skill .md file."""
    text = path.read_text(encoding="utf-8")
    m = _FRONTMATTER_RE.match(text)
    name, description, body = path.stem, "", text
    if m:
        for line in m.group(1).splitlines():
            if line.startswith("name:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("description:"):
                description = line.split(":", 1)[1].strip()
        body = text[m.end():]
    return name, description, body.strip()


def _skill_md(skill: Skill) -> str:
    return (
        f"---\nname: {skill.name}\ndescription: {skill.description}\n---\n\n"
        f"{skill.content}\n"
    )


class SkillPool:
    def __init__(
        self,
        repo_dir: str | Path = "skill_repo",
        eval_dir: str | Path = "skills",
    ):
        self.repo_dir = Path(repo_dir)   # permanent — all skills
        self.eval_dir = Path(eval_dir)   # ephemeral — sampled group only

        self.repo_dir.mkdir(parents=True, exist_ok=True)
        self.eval_dir.mkdir(parents=True, exist_ok=True)

        self._registry_path = self.repo_dir / "registry.json"
        self._skills: Dict[str, Skill] = {}
        self.generation: int = 0
        self.best_group: Dict = {}   # highest-reward group seen so far

        self._load()

    # ------------------------------------------------------------------ #
    # Load / save (repository)                                             #
    # ------------------------------------------------------------------ #

    def _load(self) -> None:
        registry: dict = {}
        if self._registry_path.exists():
            registry = json.loads(self._registry_path.read_text(encoding="utf-8"))
            self.generation = registry.get("generation", 0)
            self.best_group = registry.get("best_group", {})

        skill_states: dict = registry.get("skills", {})

        for md_file in sorted(self.repo_dir.glob("*.md")):
            sid = md_file.stem
            name, description, content = _parse_md(md_file)
            if sid in skill_states:
                skill = Skill.from_dict(skill_states[sid], content=content)
                if not skill.description and description:
                    skill.description = description
            else:
                skill = Skill(id=sid, name=name, description=description, content=content)
            self._skills[sid] = skill

    def save(self) -> None:
        """Persist skill state (V, N, logs, best_group) to registry.json."""
        payload = {
            "version": 1,
            "generation": self.generation,
            "saved_at": datetime.utcnow().isoformat(),
            "best_group": self.best_group,
            "skills": {sid: s.to_dict() for sid, s in self._skills.items()},
        }
        self._registry_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def set_best_group(self, record: Dict) -> None:
        self.best_group = record

    # ------------------------------------------------------------------ #
    # Eval workspace management                                            #
    # ------------------------------------------------------------------ #

    def stage_group(self, group: List[Skill]) -> None:
        """
        Populate skills/ with ONLY the sampled group.
        Clears any previous contents first.
        """
        # Clear eval dir
        for f in self.eval_dir.glob("*.md"):
            f.unlink()

        # Write sampled skills
        for skill in group:
            dest = self.eval_dir / f"{skill.id}.md"
            dest.write_text(_skill_md(skill), encoding="utf-8")

    def clear_eval(self) -> None:
        """Remove all files from the eval workspace."""
        for f in self.eval_dir.glob("*.md"):
            f.unlink()

    # ------------------------------------------------------------------ #
    # Skill management                                                     #
    # ------------------------------------------------------------------ #

    @property
    def skills(self) -> List[Skill]:
        return list(self._skills.values())

    def get(self, sid: str) -> Optional[Skill]:
        return self._skills.get(sid)

    def add_skill(self, name: str, description: str, content: str) -> Skill:
        """Write a new skill to repo and register it in the pool."""
        sid = self._new_id(name)
        md_path = self.repo_dir / f"{sid}.md"
        skill = Skill(id=sid, name=name, description=description, content=content)
        md_path.write_text(_skill_md(skill), encoding="utf-8")
        self._skills[sid] = skill
        return skill

    def _new_id(self, name: str) -> str:
        base = re.sub(r"[^a-z0-9_]", "_", name.lower())[:30].strip("_")
        if base not in self._skills:
            return base
        for i in range(1, 1000):
            cand = f"{base}_{i}"
            if cand not in self._skills:
                return cand
        return f"{base}_{int(datetime.utcnow().timestamp())}"

    # ------------------------------------------------------------------ #
    # Summary / display                                                    #
    # ------------------------------------------------------------------ #

    def summary(self, top_n: int | None = None) -> str:
        ranked = sorted(self.skills, key=lambda s: -s.V)
        if top_n:
            ranked = ranked[:top_n]
        lines = [
            f"Skill Repository — {len(self._skills)} skills  "
            f"(generation {self.generation})"
        ]
        header = f"  {'ID':<30} {'V':>6} {'N':>5}  description"
        lines.append(header)
        lines.append("  " + "-" * 80)
        for s in ranked:
            lines.append(
                f"  {s.id:<30} {s.V:>6.3f} {s.N:>5}  {s.description[:50]}"
            )
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._skills)

    def __repr__(self) -> str:
        return f"SkillPool(size={len(self)}, generation={self.generation})"
