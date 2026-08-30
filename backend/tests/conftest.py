import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.make_sample_db import build  # noqa: E402


@pytest.fixture(scope="session")
def sample_db(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("db") / "names.db"
    build(str(path))
    return str(path)


@pytest.fixture(autouse=True)
def use_sample_db(sample_db, monkeypatch):
    from app import config, database
    from app.services import forecast

    monkeypatch.setattr(config, "NAMES_DB_PATH", sample_db)
    database.resolve_database_path.cache_clear()
    forecast.forecast_name.cache_clear()
    yield
    database.resolve_database_path.cache_clear()
