"""The sample database must have the same shape as the built database.

The real database is never available to tests, so these assertions stand in for
it: the sample generator and the build script share their index DDL, and both
store observed rows only.
"""

import json
import sqlite3


def test_sample_database_stores_observed_rows_only(sample_db):
    conn = sqlite3.connect(sample_db)
    try:
        (padded,) = conn.execute("SELECT COUNT(*) FROM names WHERE total_count = 0").fetchone()
        (total,) = conn.execute("SELECT COUNT(*) FROM names").fetchone()
    finally:
        conn.close()
    assert total > 0
    assert padded == 0


def _plan_for(monkeypatch, call) -> str:
    """The query plan SQLite chooses for whatever SQL `call` actually runs."""
    from app import database

    executed: list[str] = []
    real_connect = database.connect

    def tracing_connect():
        conn = real_connect()
        conn.set_trace_callback(executed.append)
        return conn

    monkeypatch.setattr(database, "connect", tracing_connect)
    call()
    monkeypatch.undo()

    conn = database.connect()
    try:
        statement = next(sql for sql in executed if sql.lstrip().upper().startswith("SELECT"))
        rows = conn.execute(f"EXPLAIN QUERY PLAN {statement}").fetchall()
    finally:
        conn.close()
    return "\n".join(row[3] for row in rows)


def test_history_lookup_uses_an_index_rather_than_scanning(monkeypatch):
    from app.services import queries

    plan = _plan_for(monkeypatch, lambda: queries.get_name_history("emma", "F"))
    assert "SCAN" not in plan, plan
    # Being indexed on `sex` alone is the production defect, not the fix: it
    # searches an index but still visits every row of one sex. The plan has to
    # show the lowercased *name* constraining the lookup.
    assert "<expr>=?" in plan, plan
    # And the index must supply the year ordering, rather than SQLite sorting.
    assert "TEMP B-TREE" not in plan, plan


def test_top_names_lookup_uses_an_index_rather_than_scanning(monkeypatch):
    from app.services import queries

    plan = _plan_for(monkeypatch, lambda: queries.get_top_names("F", 2015, 10))
    assert "SCAN" not in plan, plan
    assert "sex=? AND year=?" in plan, plan


def test_forecasts_table_holds_only_eligible_names_and_excludes_history(sample_db):
    """The `forecasts` table's own shape and eligibility filtering aren't
    observable through the HTTP surface (an absent row and an HTTP 200 with
    an empty forecast list look the same to a client either way), so this
    asserts them directly against the artifact the batch produced. See
    docs/adr/0004-forecasts-as-a-build-artifact.md.
    """
    conn = sqlite3.connect(sample_db)
    try:
        rows = {
            (name, sex): json.loads(payload)
            for name, sex, payload in conn.execute("SELECT name, sex, payload FROM forecasts")
        }
    finally:
        conn.close()

    # Emma is in current use with ample history: eligible, and stored keyed
    # on the lowercased name.
    assert ("emma", "F") in rows
    payload = rows[("emma", "F")]
    assert "history" not in payload
    assert len(payload["forecast"]) == 5

    # Debra fell out of use before the final year: ineligible despite ample
    # history, so the batch must not have stored a row for her.
    assert ("debra", "F") not in rows

    # Mateo is in current use but has fewer than MIN_HISTORY_YEARS years:
    # ineligible for the opposite reason.
    assert ("mateo", "M") not in rows


def test_precompute_batch_reads_the_names_table_once_rather_than_per_name(tmp_path):
    """The batch's own decisive performance property — reading the whole
    `names` table once and grouping in memory, rather than issuing one query
    per eligible name/sex pair — isn't observable at the HTTP layer either,
    since the API never sees how the table underneath it was populated.
    """
    import scripts.precompute_forecasts as precompute_module
    from scripts.make_sample_db import build
    from scripts.precompute_forecasts import run as precompute

    path = tmp_path / "names.db"
    build(str(path))

    executed: list[str] = []
    real_connect = sqlite3.connect

    def tracing_connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        conn.set_trace_callback(executed.append)
        return conn

    precompute_module.sqlite3.connect = tracing_connect
    try:
        precompute(str(path))
    finally:
        precompute_module.sqlite3.connect = real_connect

    reads_of_names = [
        sql
        for sql in executed
        if sql.strip().upper().startswith("SELECT") and "FROM NAMES" in sql.upper()
    ]
    # One query reads the whole table, one finds the latest year — never one
    # per name/sex pair, even though the sample data has 11 name profiles.
    assert len(reads_of_names) <= 2, reads_of_names
