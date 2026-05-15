---
name: complete_is_win_with_subject
description: Find an existing 'IS WIN' alignment, then route and push a noun-text to complete it.
---

Scan the grid for an 'IS' adjacent to 'WIN' (horizontally or vertically). Treat this as an anchored pair: avoid pushing either block unless there is no alternative. Your goal is to bring a single noun-text into the open slot of that pair to form '<NOUN> IS WIN'.

Determine orientation and target slot. If 'IS' is left of 'WIN', you need the subject immediately left of 'IS'. If 'IS' is above 'WIN', you need the subject immediately above 'IS'. Choose a noun-text that is accessible and movable (not wedged or guarded by STOP unless you can disable it first).

Plan a straight push along the axis of the pair. Approach the subject-text from the side opposite the target slot and push it step-by-step toward the slot, clearing small obstacles first without nudging 'IS'/'WIN'. Use walls as braces to keep 'IS'/'WIN' fixed if you must push nearby blocks.

Stop as soon as the rule reads '<NOUN> IS WIN'. Then immediately path to any instance of that f-object (the physical object of that noun) to win, unless forming the rule already makes you WIN (in which case you may win instantly).
