# 6. The database is a published build artifact, baked in at container-build time

Date: 2026-08-31

## Status

Accepted

## Context

The built database (ADR 0003's observed-rows-only `names` table, plus ADR 0004's precomputed
`forecasts` table) is around 178 MB — small enough to bake into a container image, but the
repository still carried the 1.1 GB source database (`data/names.db`) tracked with Git LFS, and
nothing yet got either database to a deployed backend.

Every way of delivering a database to a running container has a cost the deployment cannot
absorb:

- **Downloading it at request time or at container startup is incompatible with scale-to-zero.**
  The backend is deployed to sleep when idle (parent PRD dwest1507/baby-names-app#5); a container
  that must fetch 178 MB before it can answer its first request pays that cost on every cold
  start, and a fetch that fails or is slow turns a wake-up into a visible failure.
- **A persistent volume reintroduces the always-on cost scale-to-zero exists to avoid.** Railway
  volumes are attached storage billed independent of whether the service is running; mounting one
  just to hold a read-only 178 MB file defeats the reason the backend scales to zero in the first
  place.
- **Keeping the database in the repository doesn't work at either size.** 1.1 GB in Git LFS was
  already the problem this whole slice of work exists to solve; even the pruned 178 MB artifact is
  a generated build output, not source — checking it in would mean committing a new 178 MB blob
  every time the pipeline refreshes or the forecast batch reruns.

Git LFS's own storage was ruled out for the same reason keeping the file in the repository was:
it is still something a fresh clone fetches, and it is billed storage/bandwidth for a project
targeting a free tier.

## Decision

**The built database is published to a public Hugging Face dataset. `backend/Dockerfile`
downloads it at image-build time and bakes it into the image at a fixed path. The running
container never calls Hugging Face — `NAMES_DB_REPO` is deliberately unset at runtime, so a
cache miss can never become a network call on the request path.**

- **Published as a public dataset, not private.** The underlying SSA data is itself public, so
  there is nothing to protect by gating it, and a public dataset needs no token to *read* —
  `NAMES_DB_REPO`/`NAMES_DB_FILE` are enough at runtime, with `HF_TOKEN` reserved for the
  (unused, but supported) case of a private repo. Publishing still requires a token, because
  Hugging Face requires authentication to *write* regardless of the resulting repo's visibility —
  `backend/scripts/publish_db.py` takes `HF_TOKEN` for exactly that step, not for anything the
  deployed backend does.
- **This reuses `app/database.py`'s existing resolution logic rather than a separate downloader.**
  That module already distinguished a missing file, an unresolved Git LFS pointer, and a
  non-database file, and already knew how to fall back to a Hugging Face download when configured
  — behaviour built for local development against a database too large to keep in a checkout. The
  Dockerfile's build-time download step is `python -c "from app.database import
  resolve_database_path; ..."`, invoking that same function rather than reimplementing
  `hf_hub_download` a second time. This is the same reasoning ADR 0004 applied to
  `fit_forecast`: one implementation, so a build-time and a request-time path cannot silently
  diverge.
- **The Dockerfile is three stages**, mirroring how the nietzsche-chat backend pre-downloads its
  embedding models so its running container never reaches Hugging Face either:
  - `base` installs dependencies with `uv sync --frozen --no-dev --no-install-project`, cached
    independently of application code.
  - `fetcher` extends `base` with the app code, then runs the download as a `RUN` step. The
    repo id, filename, repo type, and token are Docker build **ARGs**, not `ENV` — passed in only
    for this one `RUN` instruction's process environment, not written into the image's
    configuration.
  - `runtime` starts fresh `FROM base` (not `FROM fetcher`) and copies in only the resulting
    `/tmp/db/names.db`. Because Docker build ARGs are scoped to the stage that declares them,
    `runtime` never declares `NAMES_DB_REPO`/`HF_TOKEN` and so cannot see their values even by
    accident — the separation is structural, not a matter of remembering not to leak them,
    which is what makes "the credential cannot reach the runtime image" a property of the
    Dockerfile rather than a discipline someone has to maintain.
  - `runtime` sets `NAMES_DB_PATH` (a plain `ENV`, not a build arg — it's not a secret, and the
    running process needs to read it) to the fixed path the database was baked in at, and never
    sets `NAMES_DB_REPO`. With the local file already present, `resolve_database_path()`'s first
    check succeeds and the Hugging Face branch is simply never reached — there is no code path
    left that could turn a cache miss into a network call, because there is no configured
    upstream to call.
- **A separate `runtime`-vs-`fetcher` stage is also what keeps the image lean**, not just what
  keeps secrets out: the Hugging Face download cache and anything else the fetch touched live
  only in the `fetcher` stage's layers and are never copied forward. Measured against this
  project's own dependency set (numpy/scipy/statsmodels/pandas are not small): building `runtime`
  with a single-stage `RUN useradd && chown -R appuser /app` at the end — the pattern
  nietzsche-chat itself uses — produced a 1.33 GB image, because the recursive `chown` over a
  directory containing the already-large inherited `.venv` layer forces the whole tree to be
  copied into a new layer just to flip ownership bits. Creating the user first and using
  `COPY --chown=` on the two directories added afterward keeps the equivalent image at 908 MB —
  the same effective permissions, without duplicating the base layer.
- **The verification command (`backend/scripts/verify_db.py`, `make verify-db`) is the last gate
  before an artifact is published or shipped.** It reuses `app.database.describe_db_problem` for
  the same missing-file/LFS-pointer/non-database checks the backend itself makes, then confirms
  both `names` and `forecasts` are present and non-empty. Its reason for existing rather than
  trusting `build_db.py`/`precompute_forecasts.py` to have worked is stated plainly in the parent
  issue: a deploy with a missing or truncated artifact builds and starts happily — nothing
  touches sqlite until the first request — and then cannot answer anything. Verification catches
  that before a broken artifact is published or baked into an image, not after a visitor notices.
- **`data/names.db` is removed from the current tree and the Git LFS rule (`*.db filter=lfs...`
  in `.gitattributes`) retires with it**, per the parent PRD. Existing LFS history is left alone
  — rewriting it was explicitly out of scope — but a fresh clone no longer fetches the object,
  which was the actual cost. `.gitignore`'s `!data/names.db` carve-out, which existed only to
  keep that one file tracked, comes out too; `data/` is now build-artifact output only.

## Consequences

- **Publishing to Hugging Face and building the image are two separate, sequenced steps someone
  must run in order**: `make build-db` → `make precompute-forecasts` → `make verify-db` →
  `make publish-db` → image build. Nothing enforces that order except documentation and the fact
  that each step consumes the previous one's output; this is a deploy-time discipline, the same
  kind ADR 0004 already introduced for the precompute step.
- **`backend/scripts/publish_db.py` has never been run against a real Hugging Face endpoint.** It
  needs a write-scoped `HF_TOKEN`, which is not available to an agent in this environment. Its
  logic — refusing a missing database, refusing to run without a token, creating the dataset repo
  public and idempotently (`exist_ok=True`), uploading under the fixed filename `names.db` — is
  covered by tests that inject a fake `HfApi` (`backend/tests/test_publish_db.py`) and has not
  been exercised for real. Actually publishing is a manual, human-run step, the same way creating
  the Hugging Face dataset itself is (parent PRD, "manual, human-only steps").
- **The Dockerfile has been built and run for real in this environment, against real public
  Hugging Face repositories, but not against the project's own dataset** (which does not exist
  yet, pending the manual publish step above). Verified directly: the `base` stage installs the
  real dependency set; the `fetcher` stage, pointed at a nonexistent repo, fails with a real HTTP
  404 from the Hugging Face API (not a local error) — proving the network path is genuinely
  exercised rather than mocked; pointed at `hf-internal-testing/tiny-random-bert`'s `config.json`
  (a real, public, unauthenticated download), it correctly raises the "not a SQLite database"
  diagnostic on a file it actually fetched; pointed at the real public dataset
  `severo/test_iris_sqlite`'s `database.sqlite`, the full three-stage build succeeds, and the
  resulting image runs as the unprivileged `appuser`, serves `/api/health` with
  `{"status":"ok","database":"ok"}`, binds a `$PORT` set at `docker run` time rather than a
  hardcoded port, and continues to serve that health check with `--network none` — no network
  namespace at all — which is the strongest evidence available in this environment that the
  running container makes no outbound call. The one thing this cannot verify without the real
  publish step is the final artifact actually containing this project's `names`/`forecasts`
  schema; `make verify-db` against the resulting image's baked-in file is what closes that gap
  once a real dataset exists.
- **An empty-string `HF_TOKEN` was a live bug, not a hypothetical one.** Building the `fetcher`
  stage with no `HF_TOKEN` build arg set — the normal case for a public dataset — passed
  `token=""` to `huggingface_hub`, which sent `Authorization: Bearer ` (empty) and failed with
  `httpx.LocalProtocolError: Illegal header value`, before ever reaching Hugging Face's API. Fixed
  in `backend/app/config.py` by normalizing `HF_TOKEN` to `None` when unset or empty
  (`os.environ.get("HF_TOKEN") or None`), matching the pattern `BACKEND_SHARED_SECRET` already
  used. Without this, the "public dataset, no token required" path this whole ADR rests on would
  not have worked.
- **A fresh checkout has no database anywhere**, by design: `data/names.db` is gone from the
  tree, and `data/names.built.db` was already gitignored. Local development is unaffected because
  it never depended on the full database — `make sample-db` has always been the documented path
  (CLAUDE.md, README) — but anyone who still wants the full local pipeline output needs to
  regenerate it from `data_pipeline.ipynb` or fetch it from the published dataset themselves; the
  repository holds neither anymore.
- **The container image's final size (908 MB) is dominated by `numpy`/`scipy`/`statsmodels`/
  `pandas`, not the database** (178 MB once precomputed forecasts are included). A future
  reduction in image size would need to target the dependency set — e.g. whether `pandas` is
  still needed post-ADR-0004 — rather than anything this ADR controls.

## Related

- Parent PRD: dwest1507/baby-names-app#5
- Depends on: ADR 0003 (observed-rows-only database), ADR 0004 (forecasts as a build artifact,
  the other table `verify_db.py` checks for)
- Implements: dwest1507/baby-names-app#11
- Modeled on: the nietzsche-chat backend's `Dockerfile` (pre-downloading embedding models at
  build time so its container never calls Hugging Face at request time)
