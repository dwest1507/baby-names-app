.PHONY: help install dev dev-frontend dev-backend sample-db test lint format clean stop

help:
	@echo "Available commands:"
	@echo "  make install       - Install frontend and backend dependencies"
	@echo "  make dev           - Run both frontend and backend locally"
	@echo "  make dev-frontend  - Run frontend only (Next.js on :3000)"
	@echo "  make dev-backend   - Run backend only (FastAPI on :8000)"
	@echo "  make sample-db     - Build a small sample database for development"
	@echo "  make test          - Run frontend and backend tests"
	@echo "  make lint          - Run frontend and backend linters"
	@echo "  make format        - Auto-format frontend and backend code"
	@echo "  make stop          - Stop running dev servers"
	@echo "  make clean         - Remove caches and build artifacts"

install:
	@echo "Installing backend dependencies..."
	cd backend && uv sync
	@echo "Installing frontend dependencies..."
	cd frontend && npm install

dev-frontend:
	cd frontend && npm run dev

dev-backend:
	cd backend && uv run uvicorn app.main:app --reload --port 8000

dev:
	@echo "Starting full stack... (Press Ctrl+C to stop)"
	@$(MAKE) -j2 dev-frontend dev-backend

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
