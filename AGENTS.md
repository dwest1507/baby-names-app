# Agent guide

Baby Names Explorer: a Next.js frontend + Python FastAPI backend serving 145 years of SSA
baby name data, with trend charts, pooled-model popularity forecasts, and a Groq-powered
natural-language SQL chatbot. It was refactored from a single-file Streamlit app into this frontend/backend split.

## Commands

```bash
make install         # backend (uv sync) + frontend (npm install)
make sample-db        # build backend/data/sample_names.db for local dev (real db is 1.1GB, Git LFS)
NAMES_DB_PATH=data/sample_names.db make dev   # frontend :3000 + backend :8000 together
make dev-frontend     # Next.js only
make dev-backend      # FastAPI only (uv run uvicorn app.main:app --reload --port 8000)
make test             # backend pytest + frontend vitest
make lint             # ruff check/format --check + eslint + tsc --noEmit + prettier check
make format           # ruff --fix + format, prettier --write
make stop             # kill dev servers on :3000/:8000
```

Single test / narrower runs:

```bash
cd backend && uv run pytest tests/test_queries.py -k some_test
cd frontend && npx vitest run path/to/file.test.tsx
```

The AI chatbot needs `backend/.env` with `GROQ_API_KEY=...` to function; without it chat
endpoints return a "chatbot unavailable" error rather than failing at startup.

## Architecture

```
Browser → Next.js (:3000) → /api/[...path]/route.ts (proxy) → FastAPI (:8000) → SQLite names.db
```

- The browser never talks to FastAPI directly. `frontend/app/api/[...path]/route.ts` is a
  catch-all proxy: it allow-lists exact backend paths via regex (`ALLOWED_GET`/`ALLOWED_POST`)
  and forwards to `NAMES_API_URL` (default `http://localhost:8000`). Adding a new backend
  route requires updating this regex or the proxy will 404 it.
- `backend/app/database.py` resolves `names.db` lazily and caches the resolved path
  (`lru_cache`). It distinguishes three failure states — missing file, unresolved Git LFS
  pointer, and non-SQLite file — and if `NAMES_DB_REPO` is set, downloads a real copy from
  Hugging Face Hub at first access. All connections are opened read-only (`mode=ro`).
- `backend/app/services/chatbot.py` implements the chat feature as two Groq calls: one
  translates the question + recent history into SQL, one phrases query results as an answer.
  Generated SQL is validated in `validate_sql_query` (reads only — `SELECT` or a `WITH`
  CTE — keyword blocklist, and a `LIMIT 1000` applied by wrapping the query rather than by
  editing its text) before execution in `execute_safe_sql`. It runs on the budgeted
  connection from `database.connect_for_generated_sql`: read-only, five-second deadline,
  1 MB value cap. The budget, not the row cap, is what bounds a cartesian join or a
  recursive CTE — see `docs/adr/0008-a-resource-budget-for-generated-sql.md`. Any change to
  the SQL guardrails or the schema description (`SCHEMA_CONTEXT`) should keep the prompt and
  the validator in sync; a test asserts they agree.
- `backend/scripts/forecast/pooled.py` produces the forecasts shown on `/search`: one
  LightGBM booster per horizon, trained across every name's history at once and predicting
  every eligible name in one pass, from origin `MAX(year)` out five years. It replaced a
  per-name ARIMA fit that scored negative skill outside the top 1000 — see
  `docs/adr/0010-a-pooled-model-replaces-per-name-arima.md`. Feature extraction streams
  straight off `idx_names_name_sex_year` (ADR 0009) with no sort file and no intermediate
  artifact. Training is bounded to the most recent 40 origins, and what the boosters produce
  is not yet what the site draws: `pooled.point_forecasts` caps each path's implied growth,
  smooths it with an endpoint-preserving moving average over its log steps, and reconciles
  each (sex, horizon) slice onto the origin's total with one multiplicative factor — in that
  order, which is the only order in which the reconciled forecasts still add up.
  It is batch-only: the Dockerfile copies `app/` and not `scripts/`, so `lightgbm`
  and `scikit-learn` are dev-group dependencies and never reach the runtime image.
  `backend/app/services/forecast.py` holds only what the request path uses — the ADR 0001
  eligibility rule and the response composer, which fits nothing.
- The shaded bands are conformal, not model-derived: `pooled.strata_bands` takes the quantiles
  of the fit's own five-year log residuals within each `(popularity tier, volatility bin)`
  stratum, so a volatile name gets a wider band than a steady one at the same rank. A stratum
  with fewer than `MIN_STRATUM_ROWS` observed outcomes uses the whole population's band instead,
  and its holdout is counted into the population's coverage rather than publishing an estimate
  of its own — `pooled.band_stratum` is the one place that decides which. The `calibration` table
  is keyed by `(nominal_level, tier, volatility_bin)`, `forecasts.tier`/`volatility_bin` record
  the stratum each name is served under, and the API returns only the row matching it, so the
  chart labels a band with the coverage measured for names like this one. Tier is read at the
  row's own origin (`stream_series` carries a rank per year), which is what makes a historical
  backtest tier by historical ranks and serving tier by the newest year's. See
  `docs/adr/0011-conformal-bands-keyed-by-strata.md`.
- `backend/scripts/forecast/arima.py` is the frozen previous pipeline. Nothing in the batch
  calls it; `research/forecasting/methods.py` imports it so rounds 1-6 of the benchmark stay
  reproducible, which is why `statsmodels` and `scipy` remain dev-group dependencies.
- The pooled port is pinned numerically: `research/forecasting/make_parity_fixture.py`
  regenerates `backend/tests/fixtures/pooled_parity.json` from the research modules, and
  `backend/tests/test_forecast_pooled.py` demands the shipped code reproduce it. Regenerate it
  deliberately, only when the model is meant to change.
- Frontend pages under `frontend/app/` (`/`, `/explore`, `/search`, `/chat`) call the backend
  exclusively through `frontend/lib`'s typed API client, which hits the `/api/*` proxy — never
  fetch the backend URL directly from a component.
- The two root-level Jupyter notebooks (`data_pipeline.ipynb`, `model_exploration.ipynb`) are
  a separate, legacy data/ML pipeline (Selenium scraping, model experimentation) with its own
  `requirements.txt`; they're independent of the web app's dependency files (`backend/pyproject.toml`,
  `frontend/package.json`). Reproducible database builds are automated via `make build-db`
  (`backend/scripts/build_db.py`).

## Configuration

Backend env vars (see root README for the full table): `NAMES_DB_PATH`, `NAMES_DB_REPO`,
`NAMES_DB_FILE`, `NAMES_DB_REPO_TYPE`, `HF_TOKEN`, `GROQ_API_KEY`, `GROQ_MODEL`,
`ALLOWED_ORIGINS`. Frontend: `NAMES_API_URL` (proxy target).

## Agent skills

### Issue tracker

Issues live in GitHub Issues (`gh issue`); external PRs are not a triage surface. See `docs/agents/issue-tracker.md`.

### Triage labels

Default canonical label strings (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context — one `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
