---
name: assess_win_reachability_then_choose_make_win
description: Check if any WIN object is reachable; else switch to creating a local WIN rule.
---

First, enumerate all f-objects currently satisfying a WIN rule. Estimate reachability by checking for continuous paths that are not blocked by STOP properties or enclosed walls. If a WIN object is within a short Manhattan distance and no STOP-gated choke fully separates you, commit to a direct goto plan.

If all existing WIN objects are effectively gated (e.g., by a column of WALL IS STOP or by room separation), pivot immediately to rule crafting. Identify any nearby IS and WIN text that already form a pair or can be aligned quickly. Prefer using a noun-text that is physically closest to you and not part of the active YOU sentence.

When pivoting, keep the plan explicit: fix IS+WIN as an anchor if present, select the nearest safe noun-text, and compute the minimal push path to attach it. Avoid unnecessary exploration once the decision is made; follow through to completion before reconsidering.

Finally, sanity-check that at least one YOU rule remains active during edits. If crafting a WIN rule exposes a reachable WIN object, switch from editing to direct approach and finish the level.
