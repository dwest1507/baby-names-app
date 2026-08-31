"""Parameterized queries against the names table.

The table holds observed rows only: a row exists for a name/sex/year only if a
count was actually recorded against it. Because the source suppresses counts
below five, a missing row means "fewer than five, or none" — never "zero". See
docs/adr/0003-observed-rows-only.md.
"""

import json

from .. import database


def get_year_range() -> dict:
    conn = database.connect()
    try:
        row = conn.execute(
            "SELECT MIN(year) AS min_year, MAX(year) AS max_year FROM names"
        ).fetchone()
        return {"min_year": row["min_year"], "max_year": row["max_year"]}
    finally:
        conn.close()


def get_top_names(sex: str, year: int, limit: int) -> list[dict]:
    conn = database.connect()
    try:
        rows = conn.execute(
            """
            SELECT name, sex, year, total_count, popularity_percent, popularity_rank
            FROM names
            WHERE sex = ? AND year = ?
            ORDER BY total_count DESC
            LIMIT ?
            """,
            (sex, year, limit),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_name_history(name: str, sex: str) -> list[dict]:
    """A name's recorded years, oldest first.

    The `LOWER(name)` predicate must be written exactly as `db_schema` indexes
    it, or the planner cannot match the expression index and falls back to
    scanning one sex's several million rows.
    """
    conn = database.connect()
    try:
        rows = conn.execute(
            """
            SELECT name, sex, year, total_count, popularity_percent, popularity_rank
            FROM names
            WHERE LOWER(name) = LOWER(?) AND sex = ?
            ORDER BY year
            """,
            (name, sex),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_latest_data_year() -> int | None:
    """The newest year present in the data, read from the data itself.

    Forecast eligibility is defined against this rather than a hardcoded year,
    so next year's data refresh needs no code change.
    """
    conn = database.connect()
    try:
        row = conn.execute("SELECT MAX(year) AS newest FROM names").fetchone()
        return row["newest"]
    finally:
        conn.close()


def get_forecast(name: str, sex: str) -> dict | None:
    """The precomputed forecast blob for a name/sex, or None if there isn't one.

    A missing row means either the name was ineligible when the batch last
    ran (see docs/adr/0001-forecast-only-names-in-current-use.md), or it has
    no rows in `names` at all. `forecasts.name` is stored lowercased, matching
    how the batch (`scripts/precompute_forecasts.py`) keys it.
    """
    conn = database.connect()
    try:
        row = conn.execute(
            "SELECT payload FROM forecasts WHERE name = LOWER(?) AND sex = ?",
            (name, sex),
        ).fetchone()
        return json.loads(row["payload"]) if row else None
    finally:
        conn.close()
