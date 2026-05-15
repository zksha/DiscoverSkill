---
name: detect_stall_and_replan
description: Detect no progress and replan: switch targets or attempt a rule change.
---

Use this when repeated moves fail to reduce distance to the target or you keep bouncing/idling. Track the last few (dx, dy) distances to the current WIN target. If distance is non-decreasing for 3+ consecutive steps, or the observation repeats, you are stalled.

On stall, diagnose the cause: 
- If an X IS STOP rule blocks the corridor, pivot to disabling it (disable_stop_blockade) or route around with a new target tile. If a direct route requires passing a STOP wall, try to break or move the rule text instead of hammering the wall.
- If no physical instance of a WIN noun is reachable/visible, switch to creating a closer WIN (assemble_win_rule or create_alternate_win_condition). Prefer forming <nearby noun> IS WIN or BABA IS WIN only if it’s safe and preserves a YOU rule.

Replan loop:
- Re-identify WIN candidates (disambiguate_text_and_objects), pick the nearest reachable one, and resume with greedy_step_toward_win.
- If you stall again on the same obstacle, escalate immediately to rule manipulation rather than repeating the blocked approach.
