"""Tests for the precompute batch itself.

The batch is otherwise exercised only indirectly, through the `sample_db`
session fixture in conftest.py. These cover the properties that fixture cannot
assert: that fanning the work across processes does not change the result, and
that the calibration aggregate is derived from the stored table rather than
from whatever one invocation happened to fit.
"""

import json
import sqlite3
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

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


def test_workers_do_not_change_the_result(built, tmp_path):
    """Parallelism is throughput only.

    Each name is fitted independently from its own history, so fanning out
    must be bit-identical to running serially. This is the whole safety
    argument for defaulting the CLI to every core, and it is why `run` keeps a
    serial path rather than always constructing a pool.
    """
    serial = tmp_path / "serial.db"
    parallel = tmp_path / "parallel.db"
    serial.write_bytes(Path(built).read_bytes())
    parallel.write_bytes(Path(built).read_bytes())

    serial_result = run(str(serial), workers=1)
    parallel_result = run(str(parallel), workers=2)

    assert serial_result["eligible"] == parallel_result["eligible"]
    assert serial_result["stored"] == parallel_result["stored"]
    assert _forecasts(str(serial)) == _forecasts(str(parallel))
    assert _calibration(str(serial)) == _calibration(str(parallel))


def test_resume_skips_stored_names_but_still_calibrates_on_all_of_them(built, tmp_path):
    """Calibration must describe the whole table, not just this run's names.

    Coverage is a population statistic. When it was accumulated in memory
    during the run, a resumed batch calibrated only on the names it happened
    to refit — a non-random slice, since rows are written in name order — and
    published a coverage figure that no longer described the data. Storing
    each name's contribution and summing over the table is what removes that
    failure mode, so this test pins it.
    """
    db = tmp_path / "resume.db"
    db.write_bytes(Path(built).read_bytes())

    full = run(str(db), workers=1)
    assert full["stored"] > 1
    expected = _calibration(str(db))

    # Drop one name so a resume has exactly one name left to do.
    conn = sqlite3.connect(str(db))
    victim = conn.execute("SELECT name, sex FROM forecasts ORDER BY name LIMIT 1").fetchone()
    conn.execute("DELETE FROM forecasts WHERE name = ? AND sex = ?", victim)
    conn.commit()
    conn.close()

    resumed = run(str(db), workers=1, resume=True)

    assert resumed["eligible"] == 1
    assert resumed["resumed_skips"] == full["stored"] - 1
    assert _calibration(str(db)) == expected


def test_a_name_that_exceeds_the_timeout_is_stored_as_no_forecast(built, tmp_path):
    """Abandoning a name must land on an existing code path, not a new one.

    A timeout stores nothing for that name, which is the same state an
    ineligible name produces and which `forecast.build_response` already
    renders as an empty forecast list.
    """
    db = tmp_path / "timeout.db"
    db.write_bytes(Path(built).read_bytes())

    result = run(str(db), workers=1, timeout=1e-6)

    assert result["eligible"] > 0
    assert result["stored"] == 0
    assert result["timed_out"] == result["eligible"]
    assert _forecasts(str(db)) == {}
    assert _calibration(str(db)) == {}


def test_a_runaway_fit_is_killed_rather_than_asked_to_stop(built, tmp_path, monkeypatch):
    """The cap must survive code that never returns to the interpreter.

    An in-process alarm cannot do this: `signal.setitimer` is only delivered
    between bytecode instructions, so a fit spinning inside compiled
    statsmodels code never handles it. Simulating that here with a fit that
    ignores signals entirely is what distinguishes a real timeout from one
    that merely appears to work on well-behaved input.
    """
    import scripts.precompute_forecasts as module

    def _uninterruptible(history):
        import signal as _signal

        _signal.signal(_signal.SIGALRM, _signal.SIG_IGN)
        time.sleep(30)
        raise AssertionError("worker should have been killed")

    monkeypatch.setattr(module, "_fit_one", _uninterruptible)

    db = tmp_path / "runaway.db"
    db.write_bytes(Path(built).read_bytes())

    started = time.monotonic()
    result = module.run(str(db), workers=2, timeout=1.0)
    elapsed = time.monotonic() - started

    assert result["stored"] == 0
    assert result["timed_out"] == result["eligible"]
    assert _forecasts(str(db)) == {}
    # Each name costs about the timeout, not the full 30s sleep.
    assert elapsed < 5 + result["eligible"], f"runaways were not killed ({elapsed:.1f}s)"


def test_an_abandoned_name_is_reported_by_name(built, tmp_path, monkeypatch):
    """Which names were dropped has to be visible, not just how many.

    A name with no stored forecast renders as history-only — the same as an
    ineligible one — so silently abandoning it looks identical to it never
    having qualified.
    """
    import scripts.precompute_forecasts as module

    monkeypatch.setattr(module, "_fit_one", lambda history: None)

    db = tmp_path / "abandoned.db"
    db.write_bytes(Path(built).read_bytes())
    result = module.run(str(db), workers=1)

    assert result["timed_out"] == result["eligible"]
    assert len(result["abandoned"]) == result["eligible"]
    assert all(isinstance(n, str) and isinstance(x, str) for n, x in result["abandoned"])


def test_coverage_is_stored_per_name_and_never_served(built, tmp_path):
    """`payload` is handed to the API verbatim, so coverage must not be in it."""
    db = tmp_path / "coverage.db"
    db.write_bytes(Path(built).read_bytes())
    run(str(db), workers=1)

    conn = sqlite3.connect(str(db))
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


def test_it_migrates_a_forecasts_table_built_before_coverage_was_stored(built, tmp_path):
    """An existing artifact must not have to be rebuilt from scratch.

    `data/names.built.db` can already carry a three-column `forecasts` table
    from a previous run, and `CREATE TABLE IF NOT EXISTS` will not widen it —
    the insert would simply fail against the real artifact.
    """
    db = tmp_path / "legacy.db"
    db.write_bytes(Path(built).read_bytes())

    conn = sqlite3.connect(str(db))
    conn.execute("DROP TABLE IF EXISTS forecasts")
    conn.execute(
        "CREATE TABLE forecasts ("
        "name TEXT NOT NULL, sex TEXT NOT NULL, payload TEXT NOT NULL, "
        "PRIMARY KEY (name, sex))"
    )
    conn.commit()
    conn.close()

    result = run(str(db), workers=1)

    assert result["stored"] > 0
    assert _calibration(str(db))
