---
name: attach_subject_via_precise_push_route
description: Compute and follow the exact push route to attach subject to 'IS WIN'.
---

After selecting a subject noun-text, plan the minimal, safe sequence of pushes to place it exactly into the subject slot of the 'IS WIN' anchor without disturbing the anchor.

Anchor geometry:
- Horizontal anchor (common): target cell is immediately left of IS. Vertical anchor: target cell is immediately above IS. Keep IS and WIN fixed where possible.

Routing algorithm:
- Align axis-by-axis:
  - If the subject’s column differs from the target’s column: route to the subject’s horizontal push-side and push horizontally until columns match (ensure free space ahead).
  - If the subject’s row differs from the target’s row: route to the vertical push-side and push vertically until rows match.
- Final approach: leave one tile gap to avoid bumping IS/WIN, get to the last push-side, and push the subject into the exact target cell to form "<NOUN> IS WIN".

Safeguards:
- Do not push IS or WIN out of alignment; if they block, reposition the subject first (go around) rather than nudging the anchor. If a STOP barrier blocks the push route, pause to minimally break that STOP, then immediately resume the planned attachment.
