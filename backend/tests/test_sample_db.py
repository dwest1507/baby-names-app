"""The sample database must have the same shape as the built database.

The real database is never available to tests, so these assertions stand in for
it: the sample generator and the build script share their index DDL, and both
store observed rows only.
"""

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
