---
name: step_toward_nearest_win_safely
description: Pick the next step toward the nearest WIN f-object, avoiding STOP collisions.
---

Given a chosen target f-object and its relative offset (dx, dy), choose the axis that reduces the larger absolute component first (greedy Manhattan descent). If |dx| > |dy|, move horizontally toward the sign of dx; otherwise move vertically toward the sign of dy.

Before committing, simulate the immediate tile in that direction. If that tile contains an f-object whose noun is currently STOP (from Active rules), do not step there. Try the other axis (if it still reduces distance). If both candidate steps would collide with STOP, replan: consider a short detour or switch to rule editing (e.g., break STOP) rather than repeatedly idling.

Never chase text blocks for navigation. Ignore entries like "rule `win` …" when picking a direction. Also avoid pushes that would break "BABA IS YOU"; if the tile you’re about to push is part of an essential YOU rule, prefer an alternate approach.

After moving, verify that the distance to the target decreased. If you are not getting closer over several steps, trigger a stall response (re-evaluate the target, consider a different WIN object, or switch to rule manipulation).
