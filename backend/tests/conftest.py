import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.make_sample_db import build  # noqa: E402

# The backend requires this on every endpoint except health; the API tests use
# one client that sends it and one that omits it.
SHARED_SECRET = "test-shared-secret"


@pytest.fixture(scope="session")
def sample_db(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("db") / "names.db"
    build(str(path))
    return str(path)


@pytest.fixture(autouse=True)
def shared_secret(monkeypatch):
    from app import config

    monkeypatch.setattr(config, "BACKEND_SHARED_SECRET", SHARED_SECRET)


@pytest.fixture(autouse=True)
def empty_rate_limit_buckets():
    """Limiter state is in-process and outlives a single test, so start each
    test with empty buckets rather than whatever the previous test spent."""
    from app.limiter import reset_limits

    reset_limits()
    yield
    reset_limits()


@pytest.fixture(autouse=True)
def use_sample_db(sample_db, monkeypatch):
    from app import config, database
    from app.services import forecast

    monkeypatch.setattr(config, "NAMES_DB_PATH", sample_db)
    database.resolve_database_path.cache_clear()
    forecast.forecast_name.cache_clear()
    yield
    database.resolve_database_path.cache_clear()
