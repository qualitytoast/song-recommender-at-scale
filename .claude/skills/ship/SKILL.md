---
name: ship
description: Ship a finished piece of work to main the way this project allows (branch, tests, commit, push, PR set to merge automatically once tests pass, sync, delete the branch). Use when a step works and needs to land on main, or when the user says "ship it", "push this", "open a PR".
---

# Ship a piece of work to main

`main` is protected: never push to it. Stay inside CLAUDE.md's git limits: a branch per piece of work,
add, commit, push, a PR set to merge automatically, delete the branch after it merges. Nothing else on GitHub
(no repo settings, workflows, force pushes, rewriting history).

1. **Branch.** If on `main`, create a branch named for the work (`git checkout -b <short-name>`). Don't switch
   branches while a training or evaluation run is using the code.
2. **Test.** `.venv/bin/python -m pytest -q`. If anything fails, stop and report the failure; don't ship.
3. **Commit.** Look at `git status` first. Add only this work's files: no secrets (`.env`, keys), nothing that
   `.gitignore` covers, no unrelated changes. One short, clear commit message.
4. **Push and open the PR.** `git push -u origin <branch>`, then `gh pr create --base main --head <branch>`
   with a short body (what changed, which checks ran), then `gh pr merge <branch> --auto --merge`.
5. **Wait for the merge in the background** (don't poll in the foreground): a loop on
   `gh pr view <number> --json state -q .state` until `MERGED` (give up after ~10 minutes), then
   `gh pr checks <number>`. If the tests failed on GitHub, report the output and stop.
6. **Sync and clean up.** `git checkout main && git pull`, `git branch -d <branch>`,
   `git push origin --delete <branch>`. If the Oracle server's checkout was moved to this branch for a deploy
   (`ssh -i ~/.ssh/oracle_songrec -o StrictHostKeyChecking=accept-new ubuntu@songrec.duckdns.org`), move it back:
   `git checkout main && git pull && git branch -D <branch>` in `~/song-recommender-at-scale`.
7. **Report** the PR number, that it merged, and that the Mac (and server) are on `main`.
