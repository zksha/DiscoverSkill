---
name: plan_text_push_to_target
description: Plan minimal safe pushes to move a text block onto a specified target tile.
---

Define the exact target tile for the text block (e.g., the slot left of 'IS' in a horizontal pair). Work backward from that tile: you must stand one tile beyond the text in the push direction, so ensure there is space behind the text at every step of the path.

Chart a straight-line push route from the text's current position to the target tile. If the direct path is obstructed or the text is against an edge, first reposition it orthogonally to create the necessary backing space. Avoid pushing into corners where you cannot get around to another side; keep at least one free adjacent tile for re-approach.

Stabilize nearby critical rules while pushing. Do not push 'IS' or 'WIN' unless unavoidable; if they shift, restore their alignment before continuing. When pushing chains of blocks, use walls as braces so only the intended block moves, and verify you won’t accidentally break 'BABA IS YOU'.

Execute the planned sequence deliberately: push along the axis toward the target, re-approach around obstacles instead of forcing awkward multi-block shoves, and replan if any push changes the layout in an unexpected way.
