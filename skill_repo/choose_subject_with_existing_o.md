---
name: choose_subject_with_existing_object
description: Choose a noun-text whose matching object exists before forming '<NOUN> IS WIN'.
---

Forming a new WIN rule only helps if that noun actually has objects on the map. Before committing to build '<NOUN> IS WIN', verify that at least one corresponding f-object (e.g., fball for 'ball') exists somewhere.

Procedure:
- Enumerate all movable noun-texts you can reach. For each noun N, check whether any fN object appears in the observation. If none exist, mark that noun as dead-on-arrival and skip it for WIN.
- Among viable nouns (with existing objects), rank candidates by total cost: distance from noun-text to the left of the 'IS WIN' anchor (or to your planned IS/WIN location), plus penalties for paths blocked by STOP. Prefer plentiful/global nouns (e.g., WALL, DOOR) when ties occur.
- Commit to the top candidate and route to the correct pushing side to attach it to the IS WIN anchor. If the world changes (e.g., object destroyed), re-evaluate candidate viability before proceeding.

This avoids wasting moves on rules like 'BALL IS WIN' when no ball exists, and instead targets a noun that can actually produce a WIN condition.
