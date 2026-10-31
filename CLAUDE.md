# song-recommender-at-scale

To see where the project stands, check `git log` and `RESULTS.md`.

## How to work

- After each step that works, add and commit w/ a short, but clear commit message.
- Use subagents to speed things up without losing quality. Training and evaluation runs share the Mac's one GPU and its memory, so run those one at a time, queued back to back, not in parallel.
- When a task needs to be broken up, for example for clarification, break it up as part A, B, C, D, e.t.c. Don't label anything inside those parts with letters (subtasks, runs, options): state each plainly by what it is, e.g. "the N = 3 + 1.5x listens run", not "run C".
- Every plan starts with a TLDR: 3–6 lines covering what changes, why, the main steps, and any risk, downtime or drawbacks. Put the full plan below it.
- Avoid making new branches, worktrees or copies of the repo unless they're really needed.
- Git and GitHub: limit yourself to making a branch for each piece of work, adding, committing and pushing to it, opening a pull request set to merge automatically once the tests pass, and deleting the branch after it merges. Nothing else (repo settings, workflows, worktrees) unless I ask.
  - Also allowed, since they change nothing on GitHub: read-only commands (`git status`, `git log`, `git diff`, `gh pr view`, `gh pr checks`), and syncing my Mac after a merge (`git checkout main`, `git pull`).

## Retriever and ranker runs

A ranker starts from its retriever's saved checkpoint (`retriever = "configs/..."` in the ranker config), so its results depend on that exact file.

- Never retrain an existing retriever config. Any change to the retriever gets a new config name (and so its own run folder), so no ranker's retriever is ever overwritten.
- A new retriever gets a baseline ranker run: copy the current ranker config, change only its `retriever =` line (and scale the check intervals if the data size changed), and run it. That run is the baseline for later ranker changes on that retriever.
- Compare ranker changes on the same retriever only. Across retrievers, compare how much each ranker adds over its own retriever alone.
- When we make a change to ranker that we keep, check to see if the same can be done for retriever and vice versa.