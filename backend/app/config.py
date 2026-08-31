import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# The chatbot degrades gracefully when no key is configured, so this is optional
GROQ_API_KEY: str | None = os.environ.get("GROQ_API_KEY")
GROQ_MODEL: str = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

# Comma-separated; strip whitespace so "a.com, b.com" works as well as "a.com,b.com"
ALLOWED_ORIGINS: list[str] = [
    origin.strip()
    for origin in os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    if origin.strip()
]

REPO_ROOT = Path(__file__).parent.parent.parent
DEFAULT_DB_PATH = str(REPO_ROOT / "data" / "names.db")

NAMES_DB_PATH: str = os.environ.get("NAMES_DB_PATH", DEFAULT_DB_PATH)
NAMES_DB_REPO: str | None = os.environ.get("NAMES_DB_REPO")
NAMES_DB_FILE: str = os.environ.get("NAMES_DB_FILE", "names.db")
NAMES_DB_REPO_TYPE: str = os.environ.get("NAMES_DB_REPO_TYPE", "dataset")
# `or None` matters here, not just style: an empty string is still passed to
# huggingface_hub as `token=""`, which sends `Authorization: Bearer ` (empty)
# and fails with `httpx.LocalProtocolError: Illegal header value b'Bearer '`
# rather than the anonymous, unauthenticated request a public dataset needs.
# Caught building backend/Dockerfile with no HF_TOKEN build arg set — the
# exact "public dataset, no token required" path this project depends on.
HF_TOKEN: str | None = os.environ.get("HF_TOKEN") or None

# Presented by the frontend proxy on every backend call except the health probe.
# Unset means no secret is required, which keeps a fresh local checkout usable;
# a deployment with no secret fails closed instead, refusing every request but
# the health probe (see docs/adr/0002-shared-secret-gateway.md).
BACKEND_SHARED_SECRET: str | None = os.environ.get("BACKEND_SHARED_SECRET") or None

# Optional. Unset locally and in CI (the default), so neither ever reports to
# Sentry and neither consumes the free tier's event budget. Set on Railway once
# a Sentry project exists; see app/sentry.py, which no-ops entirely when this
# is unset.
SENTRY_DSN: str | None = os.environ.get("SENTRY_DSN") or None

# "Am I deployed?" — set APP_ENV=production explicitly, but also infer it from
# the variables Railway injects into every container it runs, so that a
# deployment cannot end up unguarded because one variable was forgotten.
APP_ENV: str = os.environ.get("APP_ENV", "development")
IS_PRODUCTION: bool = APP_ENV == "production" or any(
    os.environ.get(marker)
    for marker in ("RAILWAY_ENVIRONMENT", "RAILWAY_ENVIRONMENT_NAME", "RAILWAY_SERVICE_ID")
)
