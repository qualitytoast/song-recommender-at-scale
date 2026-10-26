# song-recommender-at-scale

To see where the project stands, check `git log` and `RESULTS.md`.

## How to work

- After each step that works, add and commit w/ a short, but clear commit message.
- Use subagents when they'd speed things up: independent pieces of work (e.g. two separate parts of a change, or reading/reviewing a large part of the code) can go to subagents in parallel. Training and evaluation runs share the Mac's one GPU and its memory, so run those one at a time, queued back to back, not in parallel.
- When a task needs to be broken up, for example for clarification, break it up as part A, B, C, D, e.t.c.
- Run the "audit" skill automatically after every training run
- Avoid making new branches, worktrees or copies of the repo unless they're really needed.
- Git and GitHub: limit yourself to making a branch for each piece of work, adding, committing and pushing to it, opening a pull request set to merge automatically once the tests pass, and deleting the branch after it merges. Nothing else (repo settings, workflows, worktrees) unless I ask.
  - Also allowed, since they change nothing on GitHub: read-only commands (`git status`, `git log`, `git diff`, `gh pr view`, `gh pr checks`), and syncing my Mac after a merge (`git checkout main`, `git pull`).

## Retriever and ranker runs

A ranker starts from its retriever's saved checkpoint (`retriever = "configs/..."` in the ranker config), so its results depend on that exact file.

- Never retrain an existing retriever config. Any change to the retriever gets a new config name (and so its own run folder), so no ranker's retriever is ever overwritten.
- A new retriever gets a baseline ranker run: copy the current ranker config, change only its `retriever =` line (and scale the check intervals if the data size changed), and run it. That run is the baseline for later ranker changes on that retriever.
- Compare ranker changes on the same retriever only. Across retrievers, compare how much each ranker adds over its own retriever alone.
- When we make a change to ranker that we keep, check to see if the same can be done for retriever and vice versa.