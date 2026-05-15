"""
Temperature-controlled skill group sampler.

Implements:
  U(s_i) = V(s_i) + c · sqrt(log(T) / (N(s_i) + 1))   (UCB exploration bonus)
  p(s_i) = softmax(U / τ)                                (temperature-controlled)
  H       = −Σ p_i log p_i                               (sampling entropy)
  τ_t     = f(H_t)                                       (entropy-adaptive temperature)

T = Σ N(s_i)  — total skill evaluations across the pool.
N + 1 pseudocount ensures less-visited skills always get a larger bonus than
more-visited ones, even when N = 0.  c controls exploration aggressiveness.
"""

from __future__ import annotations
import math
from typing import List, Tuple

import numpy as np

from .skill import Skill


def sampling_scores(skills: List[Skill], c: float = 1.0) -> np.ndarray:
    """
    UCB score: U(s_i) = V(s_i) + c · sqrt(log(T) / (N(s_i) + 1))

    T = total evaluations (sum of all N).  Using T ensures the exploration
    bonus grows with experience, revisiting under-sampled skills over time.
    N+1 pseudocount: unvisited (N=0) always gets strictly more bonus than
    visited-once (N=1), maintaining the right exploration ordering.
    """
    V = np.array([s.V for s in skills], dtype=float)
    N = np.array([s.N for s in skills], dtype=float)
    T = max(1.0, float(N.sum()))
    return V + c * np.sqrt(np.log(T) / (N + 1))


def softmax_probs(scores: np.ndarray, tau: float) -> np.ndarray:
    shifted = scores - scores.max()          # numerical stability
    exp_s = np.exp(shifted / max(tau, 1e-8))
    return exp_s / exp_s.sum()


def entropy(probs: np.ndarray) -> float:
    return float(-np.sum(probs * np.log(probs + 1e-12)))


def sample_group(
    skills: List[Skill],
    k: int,
    tau: float = 1.0,
    c: float = 1.0,
    rng: np.random.Generator | None = None,
) -> Tuple[List[Skill], np.ndarray, float]:
    """
    Sample a group of k skills without replacement.

    Returns:
        group   – list of sampled skills
        probs   – full probability distribution over all skills
        H       – Shannon entropy of that distribution
    """
    if rng is None:
        rng = np.random.default_rng()

    k = min(k, len(skills))
    scores = sampling_scores(skills, c)
    probs = softmax_probs(scores, tau)
    H = entropy(probs)

    indices = rng.choice(len(skills), size=k, replace=False, p=probs)
    group = [skills[i] for i in indices]
    return group, probs, H


# --------------------------------------------------------------------------- #
# Adaptive temperature                                                         #
# --------------------------------------------------------------------------- #

def adapt_temperature(
    tau: float,
    H: float,
    n_skills: int,
    tau_min: float = 0.1,
    tau_max: float = 5.0,
    delta: float = 0.15,
    lo_ratio: float = 0.4,
    hi_ratio: float = 0.85,
) -> float:
    """
    Entropy-aware adaptive temperature.

    H is bounded by log(n_skills) — the entropy of a uniform distribution
    over the full pool.  We target a band [lo_ratio, hi_ratio] of that max.

      H < H_lo → too concentrated → raise τ (more exploration)
      H > H_hi → too uniform     → lower τ (more exploitation)
    """
    H_max = math.log(max(n_skills, 2))
    H_lo = lo_ratio * H_max
    H_hi = hi_ratio * H_max
    if H < H_lo:
        tau = min(tau + delta, tau_max)
    elif H > H_hi:
        tau = max(tau - delta, tau_min)
    return tau
