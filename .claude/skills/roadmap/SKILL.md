---
name: roadmap
description: Keep ROADMAP.md current. Use when a piece of work on the roadmap is finished (check it and move it to Done) or when a new idea or planned task comes up (add it to TODO Now), and before shipping any work that finishes or adds a roadmap item.
---

# Update the roadmap

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

Make the roadmap change in the same branch and PR as the work itself (see `ship`).
