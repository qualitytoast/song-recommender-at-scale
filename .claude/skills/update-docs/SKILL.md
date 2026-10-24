---
name: update-docs
description: Keep RESULTS.md, README.md and ROADMAP.md current. Use after a piece of work is measured or finished, when a new idea or planned task comes up, and before shipping any work (ship calls it).
---

# Update the docs

Make every docs change in the same branch and PR as the work itself (see `ship`).

## RESULTS.md: the lab notebook

Anything measured gets an entry (a run, an evaluation, a check, a tuning grid, a deploy with timings), in the
section it belongs to, after the entries before it:
- what changed and why, and the settings;
- every number, including what didn't help;
- every run's time (training, evaluation, total);
- what was learned, and links to the run's files (`runs/...`) or commits.

Never rewrite old entries. Only correct a mistake, and say that it was corrected.

## README.md: the front page

Keep it correct, not complete. When the work changes a fact the README states, change that fact, a line or
two at most: a headline result (a row of the results table or one line), a feature of the site, how it's
served, a command in "Running it", or a diagram (Mermaid) when the architecture changes. Details go in
RESULTS.md. Most experiments don't touch the README.

Polishing it (rewording, reorganizing, choosing what to highlight, trimming) is the user's call: never restyle
or restructure it unless asked.

## ROADMAP.md

`ROADMAP.md` has two lists: **Done** (`- [x]`, oldest first) and **TODO Now** (`- [ ]`). Every entry is one
line, about the length of the others (~110 characters), in the same plain style.

**A task is finished** (merged, deployed, or measured, whatever it was meant to deliver):
1. Find its line in TODO Now (or the closest one; a task can finish one part of a broader line).
2. Change `- [ ]` to `- [x]`, and rewrite it as what was done, with its headline result, like the Done entries
   ("Song search that fixes typos ... (Meilisearch)", "... 0.2433 vs 0.1996").
3. Move it to the end of Done. If only part of a line is done, move that part and keep the rest in TODO Now.

**A new idea or planned task** (the user's, or one the user agreed to):
1. Add a one-line `- [ ]` entry to TODO Now, next to related items, saying what it is and, briefly, why or
   what it costs.
2. Don't add ideas the user hasn't agreed to; suggest them instead.

**Never** delete or rewrite an open item because plans changed without asking the user first; say what you'd
change and why.
