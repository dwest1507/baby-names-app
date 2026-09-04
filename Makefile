.PHONY: help install dev dev-frontend dev-backend sample-db build-db precompute-forecasts \
	verify-db publish-db test lint \
	format clean stop \
	frontend-quality frontend-test frontend-build backend-lint backend-test frontend-deps \
	security-audit lighthouse ci-cd

help:
	@echo "Available commands:"
	@echo "  make install                 - Install frontend and backend dependencies"
	@echo "  make dev                     - Run both frontend and backend locally"
	@echo "  make dev-frontend            - Run frontend only (Next.js on :3000)"
	@echo "  make dev-backend             - Run backend only (FastAPI on :8000)"
	@echo "  make build-db                - Build the deployable database (observed rows only, indexed)"
	@echo "  make precompute-forecasts    - Precompute forecasts into the built database"
	@echo "                                 (all cores; PRECOMPUTE_ARGS=--resume to continue a run)"
	@echo "  make verify-db [DB=path]     - Verify a built database artifact is complete before deploying"
	@echo "                                 (defaults to data/names.built.db)"
	@echo "  make publish-db REPO=org/ds  - Publish data/names.built.db to a Hugging Face dataset repo"
	@echo "                                 (requires HF_TOKEN; see backend/scripts/publish_db.py)"
	@echo "  make sample-db               - Build a small sample database for development"
	@echo "  make test                    - Run frontend and backend tests"
	@echo "  make lint                    - Run frontend and backend linters"
	@echo "  make format                  - Auto-format frontend and backend code"
	@echo "  make stop                    - Stop running dev servers"
	@echo "  make clean                   - Remove caches and build artifacts"
	@echo ""
	@echo "CI checks (same scripts GitHub Actions runs — see scripts/):"
	@echo "  make ci-cd                   - Run ALL CI checks locally"
	@echo "  make frontend-quality        - ESLint + Prettier check + TypeScript check"
	@echo "  make frontend-test           - Vitest (single run)"
	@echo "  make frontend-build          - Next.js production build"
	@echo "  make backend-lint            - Ruff check + format check"
	@echo "  make backend-test            - Pytest"
	@echo "  make security-audit          - npm audit + pip-audit"
	@echo "  make lighthouse              - Lighthouse CI budget check (needs Chrome)"
	@echo ""
	@echo "Deploys are handled by the Vercel and Railway git integrations (push to main)."

install:
	@echo "Installing backend dependencies..."
	cd backend && uv sync
	@echo "Installing frontend dependencies..."
	cd frontend && npm install

# Reinstall frontend deps when node_modules is missing (e.g. after `make clean`)
# or when package-lock.json has actually changed. The stamp records the lockfile's
# hash rather than its mtime, so `git checkout`/`git pull` touching the file does
# not trigger a needless full reinstall.
frontend-deps:
	@cd frontend && \
	  stamp=node_modules/.deps-stamp; \
	  want=$$(sha256sum package-lock.json | cut -d' ' -f1); \
	  if [ ! -d node_modules ] || [ ! -f "$$stamp" ] || [ "$$(cat "$$stamp")" != "$$want" ]; then \
	    echo "Frontend dependencies missing or stale; running npm ci..."; \
	    npm ci && printf '%s' "$$want" > "$$stamp"; \
	  fi

dev-frontend: frontend-deps
	cd frontend && npm run dev

# The backend needs no equivalent rule: `uv run` syncs .venv from uv.lock itself.
dev-backend:
	cd backend && uv run uvicorn app.main:app --reload --port 8000

dev: frontend-deps
	@echo "Starting full stack... (Press Ctrl+C to stop)"
	@$(MAKE) -j2 dev-frontend dev-backend

build-db:
	@echo "Building the deployable database from data/names.db ..."
	cd backend && uv run python scripts/build_db.py
	@echo "Built data/names.built.db — observed rows only, indexed."

precompute-forecasts:
	@echo "Precomputing forecasts into data/names.built.db across all cores."
	@echo "Expect tens of minutes on the real database. Safe to interrupt: re-run with"
	@echo "PRECOMPUTE_ARGS=--resume to fit only the names still outstanding."
	cd backend && uv run python scripts/precompute_forecasts.py $(PRECOMPUTE_ARGS)
	@echo "Forecasts stored in the forecasts table of data/names.built.db."

verify-db:
	cd backend && uv run python scripts/verify_db.py $(DB)

publish-db:
	@echo "Publishing data/names.built.db to the Hugging Face dataset repo (requires HF_TOKEN)..."
	cd backend && uv run python scripts/publish_db.py $(REPO)

sample-db:
	@echo "Building sample database at backend/data/sample_names.db ..."
	cd backend && uv run python scripts/make_sample_db.py data/sample_names.db
	@echo "Point the backend at it with: NAMES_DB_PATH=data/sample_names.db"

test:
	cd backend && uv run pytest
	cd frontend && npm test

lint:
	cd backend && uv run ruff check . && uv run ruff format --check .
	cd frontend && npm run lint && npm run typecheck && npm run format:check

format:
	cd backend && uv run ruff check --fix . && uv run ruff format .
	cd frontend && npm run format

stop:
	@echo "Stopping running servers on ports 3000 and 8000..."
	-@lsof -ti:3000 | xargs kill -9 2>/dev/null || true
	-@lsof -ti:8000 | xargs kill -9 2>/dev/null || true

clean:
	@echo "Cleaning up..."
	cd backend && rm -rf .pytest_cache .ruff_cache __pycache__ app/__pycache__
	cd frontend && rm -rf .next node_modules coverage

frontend-quality:
	./scripts/frontend-quality.sh

frontend-test:
	./scripts/frontend-test.sh

frontend-build:
	./scripts/frontend-build.sh

backend-lint:
	./scripts/backend-lint.sh

backend-test:
	./scripts/backend-test.sh

security-audit:
	./scripts/security-audit.sh

lighthouse:
	./scripts/lighthouse.sh

ci-cd: frontend-quality frontend-test frontend-build backend-lint backend-test security-audit lighthouse
	@echo ""
	@echo "ci-cd: all checks passed ✔"
