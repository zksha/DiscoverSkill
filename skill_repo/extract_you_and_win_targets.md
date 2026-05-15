---
name: extract_you_and_win_targets
description: Extract YOU objects and all WIN object types from active rules; ignore text targets.
---

Read the Active rules and build three sets: YOU-nouns, WIN-nouns, and STOP-nouns. Parse only rules of the form "<NOUN> IS YOU/WIN/STOP" from the Active rules section. This tells you what you control, what you must touch, and what blocks your movement.

Then scan the map listing for physical objects whose names start with "f" (e.g., fball, fdoor). Only these are real tiles to navigate to or interact with. Lines like "rule `win` 1 step up" are text blocks, not winning tiles. Treat rule text only as resources for rule editing, never as navigation targets.

Pick target instances as follows: if any WIN-nouns exist, select the nearest f-object whose noun matches a WIN-noun. If multiple, choose the one with the smallest Manhattan distance. Record the relative offset (dx, dy). If no WIN-nouns exist, flag the situation as make_win: you should switch to forming a WIN rule instead of navigating.

Also record STOP-nouns; you will use them to detect immediate collisions when choosing a step. Keep the current YOU-nouns so you don’t accidentally break "YOU" while pushing text.
