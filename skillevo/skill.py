"""Skill dataclass — the fundamental unit in the skill pool."""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List


@dataclass
class Skill:
    id: str
    name: str
    description: str
    content: str          # full markdown body (injected into agent prompt)
    V: float = 0.0        # EMA of credit-allocated reward: E[C_i · R_G]
    N: int = 0            # visitation count (for freshness bonus)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # Per-skill execution log
    logs: List[Dict[str, Any]] = field(default_factory=list, repr=False)

    def add_log(self, entry: Dict[str, Any]) -> None:
        self.logs.append(entry)

    # ------------------------------------------------------------------ #
    # Serialization                                                        #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "V": self.V,
            "N": self.N,
            "created_at": self.created_at,
            "logs": self.logs,
        }

    @classmethod
    def from_dict(cls, d: dict, content: str = "") -> Skill:
        s = cls(
            id=d["id"],
            name=d["name"],
            description=d.get("description", ""),
            content=content,
            V=d.get("V", 0.0),
            N=d.get("N", 0),
            created_at=d.get("created_at", datetime.utcnow().isoformat()),
        )
        s.logs = d.get("logs", [])
        return s

    def __repr__(self) -> str:
        return (
            f"Skill(id={self.id!r}, name={self.name!r}, "
            f"V={self.V:.3f}, N={self.N})"
        )
