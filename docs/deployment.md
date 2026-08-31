# Deployment

Baby Names Explorer deploys as two independently-hosted services, both deployed via their
platform's native GitHub integration rather than through CI. This document is the reproducible
record of how that's wired up. **No deployment has actually been performed from this repository
as of writing this document** — every dashboard step below was authored from the platforms' own
documentation and this repo's ADRs, not observed against a live deploy. `scripts/deploy-wizard.sh`
is the interactive companion to this document: run it to walk the manual steps one at a time and
verify what can be checked from outside once you've done them.

## Overview

| | Frontend | Backend |
|---|---|---|
| Platform | Vercel | Railway |
| Framework/runtime | Next.js 16 | Docker (Python 3.11, FastAPI) |
| Deploys from | `main`, via Vercel's GitHub integration | `main`, via Railway's GitHub integration |
| Root directory | `frontend` | `backend` |
| Build | Vercel's Next.js preset | `backend/Dockerfile` (forced — see the builder trap below) |
| Config file | `frontend/vercel.json` | `backend/railway.json` |
| Skips its own build when | only `backend/**` changed (`frontend/scripts/vercel-ignore-build.sh`) | only `frontend/**` changed (`railway.json`'s `watchPatterns`) |
| Scales | always-on (Vercel serverless functions) | to zero when idle (`railway.json`'s `sleepApplication`) |
| Health check | N/A (static/serverless) | `GET /api/health`, no secret required |
| Error reporting | none (out of scope — see parent PRD) | Sentry, only if `SENTRY_DSN` is set |
| Data source | calls the backend exclusively via `/api/*` proxy | database baked into the image at build time from a published Hugging Face dataset |

CI (`​.github/workflows/*.yml`) never deploys and holds no deploy credentials for either platform.
Its role is to gate `main` — see [ci-cd.md](ci-cd.md).

## Push-to-deploy

```
                          push to main
                                │
                 ┌──────────────┴──────────────┐
                 ▼                              ▼
        Vercel git integration          Railway git integration
                 │                              │
     ignoreCommand: unchanged?           builder: DOCKERFILE (forced,
     (frontend/scripts/                   railway.json — see trap #1)
      vercel-ignore-build.sh)                    │
                 │ skip if backend-only    watchPatterns: backend/**
                 │ changes only                  │ skip if frontend-only
                 ▼                                 changes only
        Next.js build, Root                       ▼
        Directory=frontend,               docker build (downloads the
        Framework=Next.js                  published DB at build time,
        (dashboard setting —               bakes it in — ADR 0006)
        see trap #2)                              │
                 │                                 ▼
                 ▼                         container starts, binds $PORT,
         Vercel URL serves                 healthcheck: GET /api/health
         the app; /api/* proxies                  │
         to NAMES_API_URL                          ▼
         (Railway URL) with                Railway URL serves the API,
         BACKEND_SHARED_SECRET              guarded by BACKEND_SHARED_SECRET
         attached                           on every path but /api/health
```

Browsers only ever talk to the Vercel origin. The frontend's proxy
(`frontend/app/api/[...path]/route.ts`) is the only thing that ever calls the Railway URL
directly, and it's also the only thing that ever sees `BACKEND_SHARED_SECRET` — neither the
backend's URL nor the secret reaches the browser (ADR 0002).

## First-time setup, in dependency order

Do these in order — each step's output is the next step's input. `scripts/deploy-wizard.sh`
walks steps 2–6 interactively; step 1 is deliberately not automated by the wizard (see its Stage
1 for why).

### 1. Build and publish the database artifact

The backend never builds its own database; it downloads a published copy at *container-build*
time (ADR 0006). Nothing after this step works without it existing first.

```bash
make build-db               # observed rows only, indexed — data/names.built.db
make precompute-forecasts   # slow against the real db — expect a long batch run (ADR 0004)
make verify-db               # confirms the artifact is complete before you ship it
HF_TOKEN=... make publish-db REPO=yourname/baby-names-db   # needs a write-scoped HF token
```

`HF_TOKEN` needs write access even though the resulting dataset is created public — Hugging Face
requires authentication to write regardless of the resulting repo's visibility. No token is
needed to *read* the published dataset.

### 2. Railway: the backend

New project, on your existing Railway account, from this repo, with root directory `backend`.
`backend/railway.json` sets `"builder": "DOCKERFILE"` explicitly — **confirm in the dashboard
that Railway actually built from the Dockerfile** rather than silently defaulting to its own
builder (trap #1 below). Configure the healthcheck path (`/api/health`), confirm scale-to-zero,
and set the environment variables in the [reference table](#environment-variables) below,
including `BACKEND_SHARED_SECRET` (generate it with `openssl rand -hex 32` — the wizard does this
for you).

### 3. Vercel: the frontend

New project, from the same repo, root directory `frontend`, **Framework Preset: Next.js** set
explicitly in the dashboard (trap #2 below — this can't be expressed in `vercel.json`).
`frontend/vercel.json` already configures the ignored build step. Set `NAMES_API_URL` to the
Railway URL from step 2, and `BACKEND_SHARED_SECRET` to the *same* value set on Railway.

### 4. Sentry (optional)

Create a Sentry project (platform: Python → FastAPI), copy its DSN, and set `SENTRY_DSN` on
Railway. Skippable entirely — `backend/app/sentry.py` is a complete no-op without it.

### 5. Branch protection

Protect `main` behind the checks listed in [ci-cd.md](ci-cd.md), so a red build can't reach
either platform's push-triggered deploy.

### 6. Close the loop

Open the Vercel URL, search for a name, confirm the chart and forecast load, and try the chatbot
if `GROQ_API_KEY` is set. Push a change that touches only `frontend/` and confirm (in each
platform's dashboard) that only Vercel rebuilds; push a change that touches only `backend/` and
confirm only Railway rebuilds. This is the step that actually proves the ignore-build script and
watch paths, not just that their config files parse.

## Two traps, carried over from the nietzsche-chat deployment

These are the two failure modes most likely to eat an afternoon, because both produce a build
that *looks* successful.

**Trap 1 — a platform's default builder silently wins over the Dockerfile.** Railway (and
platforms like it) will happily auto-detect a buildable Python/Node project with something like
Nixpacks even when a `Dockerfile` is present, unless the builder is forced. The build succeeds,
the service starts, and it's running the wrong thing entirely — not a broken deploy of the right
image, a *healthy* deploy of a different one (no baked-in database, wrong dependency set, etc.).
`backend/railway.json` forces `"builder": "DOCKERFILE"` specifically to prevent this, but **verify
it in the dashboard after the first deploy anyway** — config-as-code support and defaults have
moved before, and the whole point of this trap is that the symptom looks identical to success.

**Trap 2 — a framework-preset mistake yields a clean build where every route 404s.** If Vercel's
Framework Preset isn't set to Next.js (e.g. left on a generic/static preset, or misdetected
because Root Directory wasn't set to `frontend` first), the build log can complete without error
while the deployed site 404s on every route, because Vercel never wired up Next.js's routing.
This is a dashboard setting — `frontend/vercel.json` cannot express it — so it has to be checked
by hand: Project Settings → General → Framework Preset → Next.js.

## Failure-symptom tables

These are written from the platforms' documented behavior and this repo's own failure modes
(the shared-secret gateway, the health check shape, the Dockerfile's three stages), not from a
lived incident — no deploy has been run from this repository yet. Treat them as a first triage
pass, not a guarantee of the exact wording a real log will show.

### Railway (backend)

| Symptom | Likely cause | Check |
|---|---|---|
| Build succeeds, service starts, but every request 404s or the app behaves unexpectedly | Trap 1: the default builder won, not the Dockerfile | Dashboard → Deployments → build logs: does the log show `docker build` steps, or Nixpacks phases? |
| Healthcheck never passes, deploy stuck "unhealthy" | Healthcheck path misconfigured, or the container isn't binding `$PORT` | Confirm `healthcheckPath: /api/health` in Settings → Deploy; confirm the Dockerfile's `CMD` binds `${PORT:-8000}`, not a hardcoded port |
| Every request (except `/api/health`) returns `401 Unauthorized` | `BACKEND_SHARED_SECRET` unset on Railway, or drifted from Vercel's value | Compare the two platforms' env vars directly — see [secret drift](#things-easy-to-get-wrong-later) below |
| `/api/health` returns `{"status":"ok","database":"unavailable"}` | The image built without the database baked in, or with a truncated/corrupt one | Re-run `make verify-db` against the artifact before the next deploy; check the Dockerfile's `fetcher` stage log for the Hugging Face download |
| Deploy fails at the `fetcher` stage with an HTTP error from Hugging Face | The dataset doesn't exist yet, is misnamed, or `HF_TOKEN`/repo id build args are wrong | Confirm the dataset was actually published (step 1); confirm `NAMES_DB_REPO` build arg matches the published repo id exactly |
| One visitor's requests intermittently get rate-limited far below the documented ceiling | The service scaled to zero and back; in-process limiter buckets reset to empty on every sleep | Expected behavior, not a bug — see [rate limiter state](#things-easy-to-get-wrong-later) below |
| Chat endpoint always fails with a provider error | `GROQ_API_KEY` unset on Railway | Set it; without it the endpoint degrades to "chatbot unavailable" rather than 500ing, so a raw provider error usually means the key is set but invalid |
| No errors ever appear in Sentry despite real backend exceptions | `SENTRY_DSN` unset (this is also the *correct* state for local dev/CI) | Confirm `SENTRY_DSN` is actually set on the Railway service, not just locally |

### Vercel (frontend)

| Symptom | Likely cause | Check |
|---|---|---|
| Clean build log, but every route 404s | Trap 2: Framework Preset isn't Next.js, or Root Directory isn't `frontend` | Project Settings → General: confirm both |
| Every page loads, but the app can't reach any data — network errors under `/api/*` | `NAMES_API_URL` unset, wrong, or pointing at a sleeping/dead Railway service | Confirm the env var, and confirm the Railway service is actually up (curl its `/api/health` directly) |
| `/api/*` calls all return `401`/`502` from the proxy | `BACKEND_SHARED_SECRET` unset on Vercel or drifted from Railway's value | Same drift check as the Railway table above |
| A backend-only push still triggers a Vercel rebuild | `frontend/vercel.json`'s `ignoreCommand` isn't wired up, or Vercel is executing it from an unexpected working directory | Check the deploy's build log for the ignore-command's own output (`vercel-ignore-build: ...`); confirm Settings → Build and Deployment → Ignored Build Step points at the script |
| A frontend-only push doesn't rebuild Vercel at all (opposite problem) | The ignore-command is misdetecting "no changes" | Run `frontend/scripts/vercel-ignore-build.sh` locally against the same commit range and inspect its exit code (`0` = skip, nonzero = build — Vercel's convention, the inverse of a typical shell script's) |
| Preview deployment works, but the shared secret looks wrong or missing | Preview environments don't inherit "Production"-only env vars | Set `BACKEND_SHARED_SECRET` and `NAMES_API_URL` for **all** environments, not just Production — previews intentionally share the production backend (parent PRD) |
| Chat responses never arrive / always error, even though Railway's own health check is fine | Provider key issue on the backend, or the request never reached the backend at all | Check Railway logs directly; if nothing arrives there, the proxy or secret is the problem, not the chatbot |

## Everyday deploys

Push to `main` (after your PR's checks pass — see [ci-cd.md](ci-cd.md)). Both platforms deploy
automatically and independently:

- A change under `frontend/**` (with nothing under `backend/**`) triggers only Vercel.
- A change under `backend/**` (with nothing under `frontend/**`) triggers only Railway.
- A change touching both, or neither directory's watch pattern being what you expected, triggers
  both — if that surprises you, it's worth re-checking the ignore-command/watch-pattern
  configuration rather than assuming it's fine.

Preview deployments (Vercel, on every PR) point at the **same production backend**, using the
same shared secret — there is no separate staging backend or separate preview secret (deliberate,
per the parent PRD).

## Rollback

**Railway:** Dashboard → your service → Deployments → find the last known-good deployment →
"Redeploy". This re-runs that exact previously-built image; it does not rebuild from source, so
it's fast and doesn't depend on the current state of `main`.

**Vercel:** Dashboard → your project → Deployments → find the last known-good deployment → the
overflow menu → "Promote to Production" (naming varies slightly by Vercel UI version — look for
the instant-rollback action, not a redeploy). This re-serves a previously-built artifact rather
than rebuilding.

Neither rollback path touches the database artifact or the shared secret — both are independent
of the deployed code, so rolling back the app does not roll back which database version is
baked into a Railway image (that's controlled by which image is currently deployed) or require
re-entering the secret.

## Local environment

See the README's [Quick Start](../README.md#quick-start) for running the app locally
(`make install`, `make sample-db`, `NAMES_DB_PATH=data/sample_names.db make dev`). Local
development never touches Railway, Vercel, Hugging Face, or Sentry unless you deliberately point
`NAMES_DB_REPO`/`SENTRY_DSN` at them.

## Things easy to get wrong later

**The shared secret lives in two places, and drift between them fails every request without
looking like an auth problem.** `BACKEND_SHARED_SECRET` must be set to the *identical* value on
both the Railway service and the Vercel project. If they drift — one rotated, one not — every
request the proxy sends fails `hmac.compare_digest` on the backend and gets a flat `401`. That
401 looks exactly like "no secret configured" or "wrong secret entirely" from the outside; there
is no separate error for "close but not equal." Rotating the secret is therefore always a
two-place change, done together. See ADR 0002.

**CORS is explicitly not what guards the backend.** The backend keeps an `ALLOWED_ORIGINS` CORS
configuration as defence in depth, but no browser request ever reaches the backend directly —
every call arrives from the frontend's server-side proxy, which CORS has no jurisdiction over
(CORS is a rule a *browser* enforces on behalf of a page; `curl` or a server-to-server call never
consults it). The shared secret is the actual guard. Do not loosen or trust the CORS
configuration as if it were doing that job. See ADR 0002.

**Short-interval keep-warm pings defeat scale-to-zero — they are not a middle-ground cost
compromise.** This backend imports in ~1.5s and holds no large in-memory state (the database is
baked into the image, nothing to warm), so there is no cold-start problem worth solving with a
keep-warm ping in the first place. Pinging the service on any short interval to "keep it warm"
simply keeps it always-on, which is the exact cost scale-to-zero exists to avoid — there's no
"ping every N minutes" interval that reduces cost while helping latency; every interval short
enough to matter is short enough to defeat scale-to-zero. Don't add one.

**The rate limiter's in-process state is lost on every sleep.** Limiter buckets live in the
Railway container's process memory. Every scale-to-zero sleep empties every visitor's bucket, so
a determined visitor can reset their own allowance by waiting for the container to sleep and
sending a new request. This is accepted, not a bug to fix — the durable alternative (Redis or
similar) reintroduces an always-on cost, and the rate limits here are a guard against a runaway
quota bill, not a security boundary. See ADR 0002.

## Environment variables

Cross-checked against `backend/app/config.py` and `frontend/app/api/[...path]/route.ts` — this
list is exhaustive as of this document, not a partial example.

### Backend (Railway; also `backend/.env` for local dev)

| Variable | Purpose | Set where |
|---|---|---|
| `NAMES_DB_PATH` | Path to a local `names.db` | Baked in by the Dockerfile in production (`/app/data/names.db`); `backend/.env` locally |
| `NAMES_DB_REPO` | Hugging Face dataset to download from when no local copy exists | Deliberately **unset** in production — the image already has the database baked in, so this is never reached (ADR 0006). Only relevant as a Dockerfile *build arg*, not a runtime env var |
| `NAMES_DB_FILE` | Filename to fetch from that repo | Build-time only, defaults to `names.db` |
| `NAMES_DB_REPO_TYPE` | `dataset` or `model` | Build-time only, defaults to `dataset` |
| `HF_TOKEN` | Token for a private Hugging Face repo | Not needed at runtime (public dataset); needed as a *build arg* only if the dataset were private, and needed by `make publish-db` (a human-run local command, never set on Railway) |
| `GROQ_API_KEY` | Enables the AI chatbot | Railway env var; `backend/.env` locally |
| `GROQ_MODEL` | Groq model for the chatbot | Optional, defaults to `openai/gpt-oss-120b` |
| `ALLOWED_ORIGINS` | CORS origins (defence in depth — not the guard, see above) | Railway env var; defaults to `http://localhost:3000` |
| `BACKEND_SHARED_SECRET` | Required on every endpoint but `/api/health`; must match the frontend's value exactly | Railway env var **and** Vercel env var — two places, see [drift](#things-easy-to-get-wrong-later) |
| `APP_ENV` | `production` makes a missing shared secret fail closed | Railway env var, set to `production` (also inferred from Railway's own `RAILWAY_ENVIRONMENT*` variables as a backstop) |
| `SENTRY_DSN` | Reports backend exceptions to Sentry; unset is a complete no-op | Railway env var only, once a Sentry project exists (step 4); **never** set locally or in CI |

### Frontend (Vercel; also `frontend/.env.local` for local dev)

Neither variable may ever gain a `NEXT_PUBLIC_` prefix — both are read only in the server-side
proxy, so the browser learns neither the backend's URL nor the secret.

| Variable | Purpose | Set where |
|---|---|---|
| `NAMES_API_URL` | Where the proxy forwards `/api/*` calls | Vercel env var, set to the Railway URL (all environments — previews share production); defaults to `http://localhost:8000` locally |
| `BACKEND_SHARED_SECRET` | Attached to every backend call by the proxy; must match the backend's value exactly | Vercel env var (all environments); unset locally runs against a backend that also has no secret configured |

## Final checklist

Maps directly to `gh issue view 12`'s acceptance criteria.

- [ ] Frontend deployed and reachable at its Vercel URL
- [ ] Backend deployed, built from `backend/Dockerfile` (not the platform's default builder —
      verified in the Railway build log), passes its healthcheck
- [ ] Backend scales to zero when idle and serves correctly after waking (search a name after a
      period of inactivity; expect the first request to be a normal cold start, not an error)
- [ ] A frontend-only push does not trigger a Railway rebuild; a backend-only push does not
      trigger a Vercel rebuild (confirmed by watching both dashboards across two real pushes)
- [ ] Only the deployed frontend can call the backend (curling the Railway URL directly, with no
      secret, returns `401` on everything but `/api/health`)
- [ ] Backend exceptions reach Sentry once `SENTRY_DSN` is set on Railway; nothing is reported
      from local development or CI (confirmed: `SENTRY_DSN` is absent from both)
- [ ] `main` is protected behind the checks listed in [ci-cd.md](ci-cd.md)
- [ ] A preview deployment works end to end against the production backend
- [ ] `scripts/deploy-wizard.sh` walks every manual step, generates the secret, and verifies the
      result without writing to either platform or to disk
- [ ] Re-running `scripts/deploy-wizard.sh` against a working deployment reports it healthy
- [ ] This document covers first-time setup, both platform traps, failure-symptom tables,
      rollback, and this checklist
- [ ] [ci-cd.md](ci-cd.md) lists the required check names verbatim
- [ ] The README points at this document rather than duplicating its content
- [ ] `backend/.env.example` and `frontend/.env.example` list every variable above
