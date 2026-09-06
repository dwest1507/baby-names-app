# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Baby Names Explorer: a Next.js frontend + Python FastAPI backend serving 145 years of SSA
baby name data, with trend charts, ARIMA forecasts, and a Groq-powered natural-language SQL
chatbot. It was refactored from a single-file Streamlit app into this frontend/backend split.

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
- `backend/app/services/forecast.py` produces the ARIMA forecasts (confidence intervals,
  holdout validation, residual diagnostics) shown on `/search`.
- Frontend pages under `frontend/app/` (`/`, `/explore`, `/search`, `/chat`) call the backend
  exclusively through `frontend/lib`'s typed API client, which hits the `/api/*` proxy — never
  fetch the backend URL directly from a component.
- The two root-level Jupyter notebooks (`data_pipeline.ipynb`, `model_exploration.ipynb`) are
  a separate, unchanged data/ML pipeline (Selenium scraping, DB generation, model
  experimentation) with its own `requirements.txt`; they're independent of the web app's
  dependency files (`backend/pyproject.toml`, `frontend/package.json`).

## Configuration

Backend env vars (see root README for the full table): `NAMES_DB_PATH`, `NAMES_DB_REPO`,
`NAMES_DB_FILE`, `NAMES_DB_REPO_TYPE`, `HF_TOKEN`, `GROQ_API_KEY`, `GROQ_MODEL`,
`ALLOWED_ORIGINS`. Frontend: `NAMES_API_URL` (proxy target).

## Agent skills

### Issue tracker

Issues live as GitHub issues in `dwest1507/baby-names-app`, managed via the `gh` CLI.
External pull requests are not a triage surface. See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical triage roles use their default label strings (`needs-triage`,
`needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See
`docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `CONTEXT.md` plus `docs/adr/` at the repo root (neither exists yet;
they're created lazily). See `docs/agents/domain.md`.
