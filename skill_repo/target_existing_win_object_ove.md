---
name: target_existing_win_object_over_win_text
description: Prefer reachable WIN objects; ignore 'WIN' text and navigate directly to them.
---

When any object currently has the WIN property, the fastest solution is usually to reach it. Do not get distracted by 'WIN' text blocks or potential rule edits if a WIN f-object is already reachable.

Procedure:
- Extract the set of f-objects that are WIN under the active rules. For each, check reachability without breaking rules: plan a path that avoids STOP tiles and respects current constraints.
- If at least one WIN object is reachable now, commit to navigation: pick the nearest and step directly toward it (greedily or via a simple safe route), avoiding unnecessary text pushes.
- Only consider rule manipulation (e.g., breaking STOP or forming new WIN) if no WIN object is currently reachable. In that case, hand off to appropriate skills (e.g., surgical STOP break or make-WIN planning).

This prevents wasted motion toward 'WIN' text or unrelated edits in goto_win scenarios with distracting rule blocks.
