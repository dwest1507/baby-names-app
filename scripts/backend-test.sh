#!/usr/bin/env bash
# Backend tests (pytest; tests build their own sample SQLite db and mock Groq).
# Called by `make backend-test` and .github/workflows/backend-ci.yml.
set -euo pipefail
cd "$(dirname "$0")/../backend"

# Same dummy values the CI job uses; tests never call the real Groq API
export GROQ_API_KEY="${GROQ_API_KEY:-test_key}"
export ALLOWED_ORIGINS="${ALLOWED_ORIGINS:-http://localhost:3000}"

echo "==> Pytest"
uv run pytest tests/ -v

echo "backend-test: OK"
