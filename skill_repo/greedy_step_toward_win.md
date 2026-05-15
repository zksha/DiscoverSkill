---
name: greedy_step_toward_win
description: Pick the next step toward the nearest WIN object, avoiding STOP tiles.
---

This skill turns a WIN target into a single safe move. It assumes you’ve already identified at least one physical WIN object (e.g., fkey) and computed its relative offset (dx, dy) from you.

1) Determine blockers from Active rules: for every X IS STOP, mark fX as impassable. Treat stepping into such tiles as illegal for the next move.
2) Choose a preferred axis: if |dx| ≥ |dy|, prefer horizontal (move left/right toward sign(dx)); otherwise prefer vertical (up/down toward sign(dy)).
3) Evaluate the preferred step: if the destination tile is not occupied by an f-object with STOP (or an unpushable obstruction), take it. If blocked, try the other axis toward sign(dy) or sign(dx).

If both direct moves are blocked, try a sidestep that preserves or reduces total distance (e.g., perpendicular move that doesn’t step into STOP). If no safe step exists, return control so higher-level logic can replan (e.g., disable_stop_blockade or assemble_win_rule). Always recheck after each move that the new (dx, dy) strictly decreased or the blocker was bypassed.
