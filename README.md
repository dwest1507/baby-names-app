# Baby Names Explorer

A modern web app for exploring 145 years of baby name popularity from the Social Security
Administration dataset — interactive trend charts, 5-year ARIMA forecasts with confidence
intervals, and an AI chatbot that answers questions about the data in natural language.

**Stack:** Next.js 16 · React 19 · TypeScript · Tailwind CSS v4 · Recharts · Python FastAPI · statsmodels · Groq

> This app was previously a single-file Streamlit app (`app.py`). It has been refactored into
> a frontend/backend architecture with a dark, Linear-style design system matching the
> [portfolio site](https://github.com/dwest1507/portfolio).

---

## Quick Start

**Prerequisites:** Node.js 20+, Python 3.11+, [uv](https://github.com/astral-sh/uv)

```bash
# Install all dependencies
make install

# Build a small sample database (the real one is a published build artifact, see below)
make sample-db

# Run frontend + backend together
NAMES_DB_PATH=data/sample_names.db make dev
```

| Service     | URL                        |
| ----------- | -------------------------- |
| Frontend    | http://localhost:3000      |
| Backend API | http://localhost:8000      |
| API docs    | http://localhost:8000/docs |

To enable the AI chatbot, create `backend/.env`:

```
GROQ_API_KEY=your_key_here
```

---

## Project Structure

```
baby-names-app/
├── frontend/               Next.js 16 app
│   ├── app/                Pages: home, /explore, /search, /chat + /api proxy
│   ├── components/         Layout, UI primitives, recharts charts
│   └── lib/                Typed API client and formatters
├── backend/                Python FastAPI
│   ├── app/
│   │   ├── database.py     names.db resolution (local path or Hugging Face download)
│   │   ├── routes/         /api/health, /api/meta, /api/top-names, /api/names, /api/chat
│   │   └── services/       Queries, ARIMA forecasting, Groq SQL chatbot
│   ├── scripts/            DB build/publish/verify scripts, sample database generator
│   ├── Dockerfile          Bakes the database in at build time (see below)
│   └── tests/              Pytest suite (runs against a generated fixture DB)
├── data/                   Build artifacts only (gitignored — see "The Database")
├── data_pipeline.ipynb     Data download/processing + ML training notebook
├── model_exploration.ipynb Model experimentation notebook
└── Makefile                Dev automation commands
```

## Features

- **Top Names** (`/explore`) — the most popular names for any year since 1880, filterable by
  sex, as a bar chart and table
- **Name Search** (`/search`) — full popularity history for any name with current-rank stat
  tiles, a 5-year ARIMA forecast (80%/95% confidence intervals), 5-year holdout validation
  metrics (MAE/RMSE/MAPE), and residual diagnostics (Ljung–Box, Jarque–Bera, ARCH, ADF)
- **AI Chat** (`/chat`) — natural-language questions are translated to SQL by Groq, executed
  against a read-only connection with keyword guards and row caps, and phrased back as an
  answer; the generated SQL is shown with every response

## Architecture

```
Browser → Next.js (:3000)                      Python FastAPI (:8000)
            ├── Static pages                     ├── GET  /api/top-names, /api/names/{name}
            │   (home, explore, search, chat)    ├── GET  /api/names/{name}/forecast (ARIMA)
            └── /api/* (proxy) ────────────────→ ├── POST /api/chat (Groq SQL chatbot)
                                                 ├── SQLite names.db (read-only)
                                                 └── Rate limiting (slowapi)
```

The browser only ever talks to the Next.js origin; a catch-all route handler
(`frontend/app/api/[...path]/route.ts`) proxies allowed API paths to the backend
(configure the target with `NAMES_API_URL`, default `http://localhost:8000`).

## Development

```bash
make dev-frontend    # Next.js on :3000
make dev-backend     # FastAPI on :8000
make test            # pytest + vitest
make lint            # ruff + eslint + tsc + prettier check
make format          # auto-format both sides
make stop            # kill dev servers
```

## The Database

The app queries a single `names` table:

| Column               | Meaning                                             |
| -------------------- | --------------------------------------------------- |
| `name`               | Baby name                                           |
| `sex`                | `M` or `F`                                          |
| `year`               | 1880–2024                                           |
| `total_count`        | Babies registered with this name that year          |
| `popularity_percent` | Share of births for that sex/year (fraction)        |
| `popularity_rank`    | Rank within the sex/year (1 = most popular)         |

The database is not stored in the repository — it's a **published build artifact**: pruned to
observed rows only, indexed, carrying precomputed forecasts (`names` and `forecasts` tables),
built with `make build-db` + `make precompute-forecasts`, verified with `make verify-db`, and
published to a public Hugging Face dataset with `make publish-db` (see
`docs/adr/0006-database-as-published-build-artifact.md`). The backend resolves a usable copy in
this order: a local file at `NAMES_DB_PATH`, or — if unset — a download from `NAMES_DB_REPO`. It
distinguishes a missing file, an unresolved Git LFS pointer (left over from before this database
moved to Hugging Face), and a non-database file, and reports whichever it finds via
`/api/health`. Configure it with (env vars or `backend/.env`):

| Setting              | Purpose                                                      | Default         |
| -------------------- | ------------------------------------------------------------ | --------------- |
| `NAMES_DB_PATH`      | Path to a local `names.db`                                   | `data/names.db` |
| `NAMES_DB_REPO`      | Hugging Face repo to download from when no local copy exists | unset           |
| `NAMES_DB_FILE`      | Filename to fetch from that repo                             | `names.db`      |
| `NAMES_DB_REPO_TYPE` | `dataset` or `model`                                         | `dataset`       |
| `HF_TOKEN`           | Token for a private Hugging Face repo (the published dataset is public and needs none) | unset |
| `GROQ_API_KEY`       | Required for the AI chatbot                                  | unset           |
| `GROQ_MODEL`         | Groq model for the chatbot                                   | `openai/gpt-oss-120b` |
| `ALLOWED_ORIGINS`    | CORS origins for the backend (defence in depth, not the guard) | `http://localhost:3000` |
| `BACKEND_SHARED_SECRET` | Required on every backend endpoint except `/api/health`; set to the same value on the frontend | unset |
| `APP_ENV`            | `production` makes a missing shared secret fail closed        | `development`   |

For local development, `make sample-db` generates a small database with a handful of names and
plausible multi-decade trends — the full dataset has never been required for `make dev`.

## Data Pipeline & ML Notebooks

The Jupyter notebooks are unchanged from the original project:

- `data_pipeline.ipynb` downloads the SSA dataset (Selenium), computes popularity metrics,
  writes `data/names.db`, and trains ML models (Linear Regression, Random Forest, XGBoost,
  LSTM) for name popularity prediction
- `model_exploration.ipynb` contains model experimentation

Install their dependencies with `pip install -r requirements.txt` (the web app itself uses
`backend/pyproject.toml` and `frontend/package.json`).

## Deployment

- **Frontend** — any Next.js host (e.g. Vercel). Set `NAMES_API_URL` to the backend's URL.
- **Backend** — `backend/Dockerfile` downloads the published database from Hugging Face at
  *build* time (via `NAMES_DB_REPO`/`HF_TOKEN` build args) and bakes it into the image at a
  fixed `NAMES_DB_PATH`, so the running container never calls a third party — no build arg or
  token exists in the runtime image, and a cache miss on the request path is structurally
  impossible. Binds `$PORT`. See `docs/adr/0006-database-as-published-build-artifact.md` for the
  reasoning and `make verify-db` to check an artifact before deploying it.

## Data Source

Data is sourced from the
[Social Security Administration's baby names database](https://www.ssa.gov/oact/babynames/limits.html)
and updated yearly.

## License

This project is for educational purposes. Please respect the SSA's terms of use for their data.
