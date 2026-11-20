# Git and GitHub Workflow

Remote: `https://github.com/tutu-er/Topo` (cited by commit links in the study records, e.g. snapshot `e66aae90`).

## Current state of this checkout (checked 2026-11-20)

- **Work in `C:\0-workspace\Topo\repo`.** It is a partial clone (`--filter=blob:none`) on the local branch `english`, which was created from `e66aae90` (the tip of `codex/project-cleanup-20261118`). The English edition was copied over it with the robocopy command below.
- **`C:\0-workspace\Topo\Topo-English` is the pre-Git copy.** Its `.git` file is a worktree pointer to `D:/0-github_workspace/Topo/.git/worktrees/Topo-English`. On this PC `D:` is Windows install media, so Git fails in that folder. Keep it only as a backup until `english` is pushed. Its `data/` and `outputs/` were not copied, so copy them into `repo` before runs that need them.
- **Tools and settings.** Git for Windows 2.56 is installed, with the global defaults of step 2 below. The commit identity is set in `repo`'s own config, matching the existing history. GitHub CLI is not installed.
- **GitHub's default branch is `master`.** Read `main` below as `master` until it is renamed.
- **Line endings.** Some top-level files in `Topo-English` had been converted to CRLF by Windows tools; the repository keeps LF.
- `.gitignore` already keeps `data/`, `outputs/`, `*.npz`, caches and build files out of Git.
- `rnj_wzzt_dr/.gitattributes` sets `* -text`: no line-ending conversion, so the SHA-256 receipts in `PROVENANCE.json` stay valid on Windows. Do not change it.
- The original local `.git` was 1.13 GiB, so use a partial clone (below).

## One-time setup on a new machine

1. Install Git for Windows (git-scm.com). It includes Git Credential Manager, which opens a browser sign-in on the first push. Optionally install GitHub CLI (`gh`).
2. Configure identity and Windows-safe defaults:

   ```powershell
   git config --global user.name  "<your name>"
   git config --global user.email "<your GitHub noreply address>"
   git config --global core.autocrlf false
   git config --global core.longpaths true
   git config --global pull.ff only
   git config --global init.defaultBranch main
   ```

3. Clone next to this folder, without downloading every historical blob up front:

   ```powershell
   git clone --filter=blob:none https://github.com/tutu-er/Topo.git C:\0-workspace\Topo\repo
   cd C:\0-workspace\Topo\repo
   git branch -r
   git log --oneline --all -10
   ```

4. Reattach the English edition:
   - **If its branch is already on GitHub:** `git switch <that-branch>`, then copy this folder over it and run `git status`. Only real local edits should show up.
   - **If it was never pushed:** `git switch -c english e66aae90`, copy this folder over it, then commit and push.

   Copy without the stale `.git` pointer:

   ```powershell
   robocopy C:\0-workspace\Topo\Topo-English C:\0-workspace\Topo\repo /E /XD .git data outputs __pycache__ /XF .git
   ```

   If the original PC that holds `D:\0-github_workspace\Topo` is still in use, push everything from there first. On that PC, `git worktree repair` fixes moved worktrees.

## Branches

| Branch | Use |
| --- | --- |
| `main` | Stable code that passes `pytest` and `python -m rnj_wzzt_dr check` |
| `exp/<study>_<yyyymmdd>` | One study: protocol, runner, tests, summary |
| `fix/<topic>` | Bug fixes to the maintained package |

Tag each sealed study at the exact source snapshot it ran from, e.g. `study/wzzt_long_grid_20261117_v2`, and push tags with `git push origin --tags`.

## Daily cycle

```powershell
git switch main
git pull                      # fast-forward only (pull.ff=only)
git switch -c exp/probing_market_20261119
# ... edit, run ...
python -m pytest -q
python -m rnj_wzzt_dr check
git status
git diff --stat
git add <specific files>      # never `git add -A` blindly
git commit -m "probing: add hourly probe planner and protocol"
git push -u origin exp/probing_market_20261119
```

Then open a pull request into `main` (`gh pr create` or the GitHub website).

## Rules that protect the evidence

- **Commit:** code, `PROTOCOL.md`, README/RESULTS summaries, small case tables and JSON manifests.
- **Never commit:** `data/`, `outputs/`, `.npz` arrays, fitted banks, solver traces, Gurobi licence files (`gurobi.lic`) or private archives. GitHub also rejects files over 100 MB.
- Study records cite commit hashes as provenance. Never force-push `main`, never rebase or amend pushed commits, and merge study PRs with a **merge commit** (not squash), so cited hashes stay in `main`'s history.
- Lock a protocol before running: commit `PROTOCOL.md`, then run from that commit. Fixes discovered mid-run become an addendum commit, as in `EXECUTION_ADDENDUM.md`.
- Keep `data/` in a separate private store (cloud drive or external disk) with its SHA-256 manifest. Git only holds the manifest.

## Optional GitHub settings

- Protect `main`: require a pull request and a passing check.
- Add a CI workflow that installs `.[test]` (NumPy/SciPy only, no Gurobi) and runs `python -m rnj_wzzt_dr check`. Solver tests stay local because they need a licence.
