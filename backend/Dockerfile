# Production image for the FastAPI backend (Railway).
#
# The database is downloaded from Hugging Face and baked into the image at
# BUILD time, mirroring how the nietzsche-chat backend pre-downloads its
# embedding models, so the running container never calls a third party. A
# cache miss must never become a network call on the request path. See
# docs/adr/0006-database-as-published-build-artifact.md.
#
# Three stages:
#   base     - installs dependencies with uv (cached independently of app code)
#   fetcher  - extends base with app code, then downloads the database using
#              app/database.py's own resolution logic, driven by build ARGs
#   runtime  - a fresh copy of base plus app code and *only* the resulting
#              database file, so the Hugging Face download cache, the
#              repo/token build args, and anything else fetching touched
#              never reach the image that actually runs.

FROM python:3.11-slim AS base

# uv, for fast, reproducible installs from uv.lock.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies before app code, so an app-only change doesn't invalidate this
# (slow: numpy/scipy/statsmodels) layer.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

ENV PATH="/app/.venv/bin:$PATH"


FROM base AS fetcher

COPY app ./app

# Build-time-only. These are Docker build ARGs, not ENV, so they never land
# in the runtime image's environment even though this stage inherits from
# `base` — an ARG's value is not carried into a later `FROM base AS runtime`
# unless that stage repeats the ARG line itself, which the runtime stage
# below deliberately does not.
ARG NAMES_DB_REPO
ARG NAMES_DB_FILE=names.db
ARG NAMES_DB_REPO_TYPE=dataset
ARG HF_TOKEN

RUN mkdir -p /tmp/db && \
    NAMES_DB_PATH=/tmp/db/names.db \
    NAMES_DB_REPO=${NAMES_DB_REPO} \
    NAMES_DB_FILE=${NAMES_DB_FILE} \
    NAMES_DB_REPO_TYPE=${NAMES_DB_REPO_TYPE} \
    HF_TOKEN=${HF_TOKEN} \
    python -c "\
from app.database import resolve_database_path; \
import shutil; \
shutil.copy(resolve_database_path(), '/tmp/db/names.db')"


FROM base AS runtime

# Created before the app-layer COPYs below, and those use --chown, rather
# than a single `RUN chown -R appuser /app` afterwards: a recursive chown
# over a directory that already contains the inherited `.venv` layer
# (numpy/scipy/statsmodels) forces the whole thing to be copied up into a
# new layer just to flip ownership bits, nearly doubling the image size.
# Measured: 907 MB with this approach vs. 1.33 GB with `chown -R`.
RUN useradd --create-home appuser

COPY --chown=appuser:appuser app ./app
COPY --from=fetcher --chown=appuser:appuser /tmp/db/names.db /app/data/names.db

# The explicit local path the database was baked in at. NAMES_DB_REPO is
# deliberately unset here: with the file already present, resolve_database_path()
# never reaches the Hugging Face branch, so no build arg or token needs to
# (or does) exist in this stage.
ENV NAMES_DB_PATH=/app/data/names.db

USER appuser

EXPOSE 8000

# Railway (and most platforms) provide $PORT at runtime; shell form so the
# variable expands rather than being passed to uvicorn literally.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
