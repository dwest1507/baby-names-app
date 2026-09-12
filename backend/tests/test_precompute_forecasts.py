"""Tests for the precompute batch itself.

The batch is otherwise exercised only indirectly, through the `sample_db`
session fixture in conftest.py. These cover the properties that fixture cannot
assert: that a published artifact is reproducible, that the extraction reads
the index rather than sorting, that one fit serves every name, and that the
coverage aggregate is derived from the stored table rather than from whatever
one invocation happened to fit.
"""

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.forecast import pooled  # noqa: E402
from scripts.make_sample_db import build  # noqa: E402
from scripts.precompute_forecasts import run  # noqa: E402


def _forecasts(db_path: str) -> dict:
    conn = sqlite3.connect(db_path)
    try:
        return {
            (name, sex): payload
            for name, sex, payload in conn.execute("SELECT name, sex, payload FROM forecasts")
        }
    finally:
        conn.close()


def _calibration(db_path: str) -> dict:
    conn = sqlite3.connect(db_path)
    try:
        return {
            level: (coverage, n)
            for level, coverage, n in conn.execute(
                "SELECT nominal_level, empirical_coverage, n FROM calibration"
            )
        }
    finally:
        conn.close()


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("precompute") / "names.db"
    build(str(path))
    return str(path)


def _copy(built: str, tmp_path, name: str) -> str:
    target = tmp_path / name
    target.write_bytes(Path(built).read_bytes())
    return str(target)


def test_two_runs_over_the_same_data_produce_identical_forecasts(built, tmp_path):
    """A published artifact has to be reproducible.

    Boosting subsamples rows and columns, and LightGBM builds its histograms
    in parallel, so nothing about the fit is reproducible by default — it is
    reproducible because the batch pins a seed and a thread count. Without
    both, rebuilding the same database twice would publish two different sets
    of numbers and there would be no way to tell a real model change from
    noise.

    Both runs go through the CLI, in separate processes. Calling `run` twice
    in one process would pass even with a seed drawn at import time, since
    both calls would draw the same one — it is exactly the run-to-run case
    that `make precompute-forecasts` is, and the only one worth pinning.
    """
    first = _copy(built, tmp_path, "first.db")
    second = _copy(built, tmp_path, "second.db")

    for db in (first, second):
        completed = subprocess.run(
            [sys.executable, "scripts/precompute_forecasts.py", db, "--threads", "2"],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr

    assert _forecasts(first) == _forecasts(second)
    assert _calibration(first) == _calibration(second)


def test_every_eligible_name_is_forecast_from_one_pooled_fit(built, tmp_path):
    """One model, every name — not one model per name.

    The count that matters is that the set of names stored is exactly the set
    the eligibility rule admits at the newest year, produced without ever
    looking at a name in isolation.
    """
    db = _copy(built, tmp_path, "pooled.db")
    result = run(db)

    conn = sqlite3.connect(db)
    try:
        (newest,) = conn.execute("SELECT MAX(year) FROM names").fetchone()
        eligible = {
            (name.lower(), sex)
            for name, sex in conn.execute(
                "SELECT name, sex FROM names GROUP BY LOWER(name), sex "
                "HAVING COUNT(*) >= 10 AND MAX(year) = ?",
                (newest,),
            )
        }
    finally:
        conn.close()

    assert eligible
    assert set(_forecasts(db)) == eligible
    assert result["stored"] == len(eligible)
    assert result["origins"]["production"] == newest


def test_extraction_reads_the_index_and_sorts_nothing(built):
    """The whole feature build is one indexed pass over `names`.

    `idx_names_name_sex_year` (ADR 0009) already stores rows in exactly the
    order the extraction wants them, so SQLite can hand them back grouped and
    ordered with no temporary B-tree and no sort file. Losing that — by
    reordering the query, or by dropping the index — turns a sub-second scan
    into a spill to disk on 11 million rows, which is the failure this pins.
    """
    conn = sqlite3.connect(built)
    try:
        plan = "\n".join(row[3] for row in conn.execute(f"EXPLAIN QUERY PLAN {pooled.SERIES_SQL}"))
    finally:
        conn.close()

    assert "idx_names_name_sex_year" in plan, plan
    assert "TEMP B-TREE" not in plan, plan


def test_forecasts_cover_only_years_that_have_not_happened_yet(built, tmp_path):
    """2025 is history, not a forecast point.

    The artifact this replaces carried forecasts for 2025-2029 produced before
    2025 was observed, so the search chart held two different values for 2025
    and the forecast overwrote the record. A forecast must start the year
    after the newest observation.
    """
    db = _copy(built, tmp_path, "years.db")
    run(db)

    conn = sqlite3.connect(db)
    try:
        (newest,) = conn.execute("SELECT MAX(year) FROM names").fetchone()
    finally:
        conn.close()

    stored = _forecasts(db)
    assert stored
    for payload in stored.values():
        years = [point["year"] for point in json.loads(payload)["forecast"]]
        assert years == list(range(newest + 1, newest + 6))


def test_no_forecast_or_band_edge_is_negative(built, tmp_path):
    """A share cannot be negative, and nothing clamps one any more.

    The forecast is the origin's share times the exponential of a predicted
    log growth, and each band edge is that times another exponential, so every
    published number is positive by construction. The old pipeline needed a
    `max(x, 0)` on every field because ARIMA's additive intervals could reach
    below zero; removing that clamp is only safe if this holds.
    """
    db = _copy(built, tmp_path, "positive.db")
    run(db)

    stored = _forecasts(db)
    assert stored
    for payload in stored.values():
        for point in json.loads(payload)["forecast"]:
            assert point["lo95"] <= point["lo80"] <= point["hi80"] <= point["hi95"]
            assert point["lo95"] > 0


def test_validation_scores_the_five_years_before_the_newest_one(built, tmp_path):
    """The holdout has to be fully observed, or it is not a holdout.

    Training at five years back and scoring against what actually happened is
    what makes the search page's predicted-against-actual table checkable
    against recorded history.
    """
    db = _copy(built, tmp_path, "validation.db")
    result = run(db)

    conn = sqlite3.connect(db)
    try:
        (newest,) = conn.execute("SELECT MAX(year) FROM names").fetchone()
    finally:
        conn.close()

    assert result["origins"]["holdout"] == newest - 5
    assert result["validated"] > 0

    validations = [json.loads(payload)["validation"] for payload in _forecasts(db).values()]
    scored = [v for v in validations if v is not None]
    assert scored
    for validation in scored:
        assert [point["year"] for point in validation["points"]] == list(
            range(newest - 4, newest + 1)
        )


def test_coverage_is_stored_per_name_and_never_served(built, tmp_path):
    """`payload` is handed to the API verbatim, so coverage must not be in it."""
    db = _copy(built, tmp_path, "coverage.db")
    run(db)

    conn = sqlite3.connect(db)
    try:
        rows = conn.execute(
            "SELECT payload, coverage_hits, coverage_n FROM forecasts WHERE coverage_n IS NOT NULL"
        ).fetchall()
    finally:
        conn.close()

    assert rows, "expected at least one name with a holdout backtest"
    for payload, hits, counts in rows:
        validation = json.loads(payload)["validation"]
        if validation is not None:
            assert "coverage" not in validation
        assert json.loads(hits).keys() == json.loads(counts).keys()


def test_it_replaces_a_forecasts_table_left_by_the_previous_pipeline(built, tmp_path):
    """An existing artifact must not have to be rebuilt from scratch.

    `data/names.built.db` carries a `forecasts` table from the ARIMA batch —
    three columns wide, and holding forecasts for years that have since been
    observed. `CREATE TABLE IF NOT EXISTS` will not widen it, and leaving its
    rows in place would serve stale years alongside the new ones.
    """
    db = _copy(built, tmp_path, "legacy.db")

    conn = sqlite3.connect(db)
    conn.execute("DROP TABLE IF EXISTS forecasts")
    conn.execute(
        "CREATE TABLE forecasts ("
        "name TEXT NOT NULL, sex TEXT NOT NULL, payload TEXT NOT NULL, "
        "PRIMARY KEY (name, sex))"
    )
    conn.execute(
        "INSERT INTO forecasts (name, sex, payload) VALUES ('zzz-retired', 'F', '{}')",
    )
    conn.commit()
    conn.close()

    result = run(db)

    assert result["stored"] > 0
    assert ("zzz-retired", "F") not in _forecasts(db)
    assert _calibration(db)


def test_the_model_card_is_stored_once_rather_than_per_name(built, tmp_path):
    """One pooled model means one description of it.

    Copying the card into every payload would repeat the same few hundred
    bytes ~24,700 times in the published artifact and leave room for two rows
    to disagree about what produced them.
    """
    db = _copy(built, tmp_path, "card.db")
    run(db)

    conn = sqlite3.connect(db)
    try:
        cards = conn.execute("SELECT payload FROM model_card").fetchall()
    finally:
        conn.close()

    assert len(cards) == 1
    card = json.loads(cards[0][0])
    assert card["features"] == list(pooled.FEATURES)
    assert card["training_rows"] > 0
    for payload in _forecasts(db).values():
        assert "model" not in json.loads(payload)


def test_the_published_forecasts_add_up_to_the_share_each_sex_held(built, tmp_path):
    """Reconciliation reaches the artifact, not just the module that does it.

    Shares within a sex sum to a fixed total, so the forecasts have to as
    well. The batch's own point stack smooths each path and then scales each
    (sex, horizon) slice onto the total observed at the origin; if either step
    were dropped on the way into `forecasts`, this is where it would show.
    """
    db = _copy(built, tmp_path, "adds-up.db")
    run(db)

    conn = sqlite3.connect(db)
    try:
        (newest,) = conn.execute("SELECT MAX(year) FROM names").fetchone()
        origin_share = {
            (name.lower(), sex): percent
            for name, sex, percent in conn.execute(
                "SELECT name, sex, popularity_percent FROM names WHERE year = ?", (newest,)
            )
        }
    finally:
        conn.close()

    totals: dict[str, list[float]] = {}
    targets: dict[str, float] = {}
    for (name, sex), payload in _forecasts(db).items():
        points = json.loads(payload)["forecast"]
        # The constraint is checked over the years actually published — the
        # five after the origin, which on the 2025 database is 2026-2030.
        assert [point["year"] for point in points] == list(range(newest + 1, newest + 6))
        means = [point["mean"] for point in points]
        totals[sex] = [a + b for a, b in zip(totals.get(sex, [0.0] * 5), means, strict=True)]
        targets[sex] = targets.get(sex, 0.0) + origin_share[(name, sex)]

    assert totals
    for sex, forecast_total in totals.items():
        for value in forecast_total:
            assert value == pytest.approx(targets[sex], rel=1e-9)


def test_the_model_card_reports_the_bounded_training_window(built, tmp_path):
    """The card describes the fit that happened, window included."""
    db = _copy(built, tmp_path, "window.db")
    result = run(db)

    conn = sqlite3.connect(db)
    try:
        (payload,) = conn.execute("SELECT payload FROM model_card WHERE id = 1").fetchone()
        (newest,) = conn.execute("SELECT MAX(year) FROM names").fetchone()
    finally:
        conn.close()

    card = json.loads(payload)
    assert card == result["model"]
    assert card["training_origins"] == len(pooled.training_origins(newest))
    assert card["training_origins"] <= pooled.TRAIN_WINDOW
