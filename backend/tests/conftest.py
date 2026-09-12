import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.make_sample_db import build  # noqa: E402
from scripts.precompute_forecasts import run as precompute_forecasts  # noqa: E402

# The backend requires this on every endpoint except health; the API tests use
# one client that sends it and one that omits it.
SHARED_SECRET = "test-shared-secret"


@pytest.fixture(scope="session")
def sample_db(tmp_path_factory) -> str:
    """The sample database, with forecasts precomputed against it.

    The batch (scripts/precompute_forecasts.py) has no test seam of its own —
    running it here means every API test exercises precomputed data, exactly
    as the deployed app would serve it. See
    docs/adr/0004-forecasts-as-a-build-artifact.md.
    """
    path = tmp_path_factory.mktemp("db") / "names.db"
    build(str(path))
    precompute_forecasts(str(path))
    return str(path)


@pytest.fixture(scope="session")
def stratified_db(tmp_path_factory) -> str:
    """A corpus big enough for the band strata to actually exist.

    The sample database has nine eligible names, so every
    `(popularity tier, volatility bin)` cell in it is far below
    `pooled.MIN_STRATUM_ROWS` and every band correctly falls back to the
    population's — which means it cannot exercise conditioning at all. This
    builds a synthetic corpus instead: names spanning a wide range of
    year-to-year wobble, with rank varying independently of it, so each cell
    fills and each gets a band of its own.

    Built from a fixed seed, so it is regenerable rather than magic. See
    docs/adr/0011-conformal-bands-keyed-by-strata.md.
    """
    import numpy as np

    from app import db_schema

    path = tmp_path_factory.mktemp("stratified") / "names.db"
    rng = np.random.default_rng(4)
    years = np.arange(1940, 2026)
    # Ranks that land in three different tiers, cycled independently of the
    # wobble so tier and volatility bin do not stand in for each other.
    ranks = (50, 500, 3000)

    rows = []
    for i in range(630):
        jitter = 0.01 + 0.35 * (i / 630)
        noise = np.cumsum(rng.normal(0.0, jitter, len(years)))
        values = 10 ** rng.uniform(-5.0, -2.5) * np.exp(noise - noise.mean())
        name, sex = f"name{i:04d}", "F" if i % 2 else "M"
        rank = ranks[i % len(ranks)]
        for year, value in zip(years, np.maximum(values, 1e-7), strict=True):
            rows.append((name, sex, int(value * 1_800_000), int(year), float(value), rank))

    conn = sqlite3.connect(str(path))
    conn.execute(db_schema.CREATE_TABLE)
    conn.executemany("INSERT INTO names VALUES (?, ?, ?, ?, ?, ?)", rows)
    db_schema.create_indexes(conn)
    conn.commit()
    conn.close()

    precompute_forecasts(str(path))
    return str(path)


@pytest.fixture
def use_stratified_db(stratified_db, monkeypatch):
    """Point the API at `stratified_db` instead of the sample database."""
    from app import config, database

    monkeypatch.setattr(config, "NAMES_DB_PATH", stratified_db)
    database.resolve_database_path.cache_clear()
    yield
    database.resolve_database_path.cache_clear()


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

    monkeypatch.setattr(config, "NAMES_DB_PATH", sample_db)
    database.resolve_database_path.cache_clear()
    yield
    database.resolve_database_path.cache_clear()
