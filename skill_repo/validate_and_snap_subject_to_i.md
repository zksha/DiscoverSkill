---
name: validate_and_snap_subject_to_is_win
description: Validate subject has object, then push it into the slot left of 'IS WIN'.
---

When an intact horizontal "IS WIN" exists, treat it as an anchor. Define the target socket as the tile immediately left of the "IS". The goal is to place a noun-text into that socket without disturbing the anchor.

Validate the subject before committing: scan noun-texts and discard any whose matching object (f-thing) is absent on the map. Prefer the closest, pushable noun-text with a clearish route to the socket. If you had previously chosen an invalid subject (e.g., BALL with no fball), immediately switch to a valid one.

Plan the push sequence to the socket. If the noun and socket share a row, approach from the noun’s left/right so that pushes move it horizontally toward the socket. If rows differ, first align vertically: get above/below the noun to push it up/down onto the socket’s row, then switch to horizontal pushes. Always route to the correct push side (behind the noun along the intended movement), and avoid contacting or pushing "IS" or "WIN". Use nearby walls as temporary stoppers when helpful; if a single "X IS STOP" blocks the corridor, perform one minimal nudge to break it, then resume the plan.

Execute as a loop: navigate to the required side, make one push, re-evaluate alignment, repeat until the noun occupies the socket and the rule forms. The moment "[NOUN] IS WIN" is active, stop editing and go directly to the nearest WIN object to finish.
