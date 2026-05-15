---
name: go_to_reachable_win_before_editing
description: If any WIN object is reachable now, navigate to it before editing rules.
---

Before manipulating any text, quickly scan the active rules for all WIN nouns and locate their corresponding objects. If a WIN object is present and a path exists that avoids STOP/solid barriers, prioritize direct navigation.

Plan a short, safe route: step toward the nearest WIN object using Manhattan distance while never stepping into STOP-blocked tiles. Ignore distractor text and unused word blocks—movement is faster and safer than rearrangement when a clean path exists.

If no WIN object is reachable (blocked by STOP or enclosure), then switch to a rule-edit plan (e.g., break STOP minimally or assemble "<NOUN> IS WIN"). This check-first approach prevents wasted moves on text when a walk-to-win is already available.
