# Baby Names Explorer API

FastAPI backend serving the baby names dataset: top-name queries, per-name
history, ARIMA popularity forecasts, and a Groq-powered natural-language
chatbot that translates questions into guarded SQL.

```bash
uv sync                                        # install dependencies
uv run python scripts/make_sample_db.py        # build a small dev database (optional)
NAMES_DB_PATH=data/sample_names.db uv run uvicorn app.main:app --reload --port 8000
```

See the repository root README for configuration and deployment details.
