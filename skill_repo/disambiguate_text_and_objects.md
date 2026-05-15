---
name: disambiguate_text_and_objects
description: Separate text from objects; target f-objects for WIN, text only for rule edits.
---

In Baba Is You logs, text blocks appear as entries like rule `key`, rule `is`, rule `win`. Physical objects appear with an f-prefix, such as fkey, fdoor, fwall. Only physical objects can be touched to satisfy WIN; text is used solely to change rules.

Procedure:
- Read Active rules and collect all nouns with the WIN property (e.g., key, flag, door).
- In the Objects list, search only for physical instances of those nouns (fkey, fflag, fdoor). Ignore rule tokens when your goal is to touch WIN.
- If no physical instance of any WIN noun exists or is visible, pivot to a rule-creation plan (e.g., assemble_win_rule or manipulate_rule_definitions) rather than chasing rule `win` or rule `key`.

Heuristics:
- Never path toward rule `win` expecting victory; it won’t win on contact. Path only toward f-nouns that currently have WIN.
- When planning edits, invert the filter: seek nearby rule blocks (rule `noun`, rule `is`, rule `win`) and treat f-objects as potential targets to end on after edits are made.
