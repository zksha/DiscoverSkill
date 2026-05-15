---
name: route_to_push_side
description: Route to the correct side of a block to push in desired direction.
---

Many failures come from standing on the wrong side of a target. This skill plans to the exact adjacent cell from which a push in the desired direction is possible, then executes the push.

Procedure: Choose the desired push direction for the target block (text or object). Compute the required approach cell (opposite the push direction: to push up, stand below; to push right, stand left). Find a path to that approach cell that avoids STOP and preserves "... IS YOU". If the approach cell is unreachable, consider an alternate push direction or first remove the obstacle (e.g., break an IS STOP sentence).

Apply iteratively for multi-push maneuvers: before each push, ensure the destination cell for the target is free and won’t create an unwanted rule break. Prefer moves that do not disturb "BABA IS YOU" unless an alternate YOU is already secured.
