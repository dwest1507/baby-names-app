# CI/CD

This app's CI never deploys and holds no deploy tokens. Deploys happen entirely through the
Vercel and Railway git integrations reacting to pushes on `main` (see
[deployment.md](deployment.md)). CI's only job is to be the gate those platforms deploy behind:
`main` should be protected so that a push whose checks would fail can't reach either platform's
production deploy.

This document exists to hold the exact, copy-pasteable check names GitHub's branch protection UI
needs. They are read directly from `.github/workflows/*.yml` as of the commit that added this
document — re-check them if a workflow file's `name:` fields change, since GitHub matches
required checks by name, not by job id.

## Required status checks, verbatim

Configure these at **Settings → Branches → Branch protection rules → main → Require status
checks to pass before merging**, searching for each name below and adding it.

| Check name (verbatim) | Workflow / job | Source |
|---|---|---|
| `Ruff lint & format` | `backend-ci.yml` → `lint` | Backend lint gate |
| `Pytest` | `backend-ci.yml` → `test` | Backend test suite |
| `Lint, format & types` | `frontend-ci.yml` → `quality` | ESLint + Prettier + `tsc --noEmit` |
| `Unit tests` | `frontend-ci.yml` → `test` | Vitest |
| `Production build` | `frontend-ci.yml` → `build` | `next build` |
| `Lighthouse CI` | `lighthouse.yml` → `lighthouse` | Performance/accessibility/SEO budget |
| `CodeQL (javascript-typescript)` | `security.yml` → `codeql` (matrix) | CodeQL analysis, JS/TS |
| `CodeQL (python)` | `security.yml` → `codeql` (matrix) | CodeQL analysis, Python |
| `Secret scanning (gitleaks)` | `security.yml` → `secret-scan` | gitleaks |
| `npm audit (frontend)` | `security.yml` → `npm-audit` | Frontend production-dependency audit |
| `pip-audit (backend)` | `security.yml` → `pip-audit` | Backend locked-dependency audit |

The two `CodeQL (...)` names come from `security.yml`'s job `name: 'CodeQL (${{
matrix.language }})'` expanded against `matrix.language: [javascript-typescript, python]` — GitHub
shows the job's `name:` field with the matrix value substituted in as the check name, so these are
the exact two check names that appear on a PR, not a guess.

### Not required on push: Dependency review

| Check name (verbatim) | Workflow / job | Why it's excluded above |
|---|---|---|
| `Dependency review` | `security.yml` → `dependency-review` | Guarded by `if: github.event_name == 'pull_request'` in the workflow — it never runs on a push to `main`, so GitHub can never see it as "passed" outside a PR context. Requiring it as a push-triggered check would permanently block merges (or, on some GitHub configurations, block indefinitely since the check would never report). |

Still worth requiring it as a **pull request** check even though it can't be a **push**-required
one — the two are configured the same way in the branch protection UI (both under "require status
checks"), GitHub simply never evaluates it outside a PR event, so treat this line as informational
rather than as an instruction to omit it from the required-checks list.

## Why this matters for deployment

Both Vercel and Railway deploy from `main` on every push, independent of GitHub Actions — CI does
not gate their deploys directly. Branch protection is what gates them indirectly: if `main` can
only be updated by a change that passed every check above, then anything the platforms pick up
from `main` already passed CI by construction. Skipping branch protection (e.g. leaving `main`
directly pushable) means CI is decorative — a red build can still land on `main` and both
platforms will deploy it.

## Configuring it

1. GitHub repo → **Settings → Branches**.
2. **Add branch protection rule**, branch name pattern `main`.
3. Enable **Require status checks to pass before merging**.
4. Search for and add each of the eleven check names in the table above (all except `Dependency
   review`).
5. Consider also enabling **Require branches to be up to date before merging** and applying the
   rule to administrators, so the protection can't be quietly bypassed.

`scripts/deploy-wizard.sh` walks this step interactively and, if the `gh` CLI is installed and
authenticated, reads back the configured required checks so you can compare them against this
list without leaving the terminal.
