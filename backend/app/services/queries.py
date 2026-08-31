"""Parameterized queries against the names table.

The table is padded: the pipeline cross-joins every name/sex pair with every
year and zero-fills the gaps, so most of its rows are fabricated rather than
observed. Every query here filters to observed rows so the rest of the app
never sees the padding. A surviving row has a count of at least five — the
source suppresses anything smaller — so a missing row means "fewer than five,
or none", never "zero".
"""

from .. import database

# A row is an observation only if a count was actually recorded against it.
OBSERVED = "total_count > 0"


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
            f"""
            SELECT name, sex, year, total_count, popularity_percent, popularity_rank
            FROM names
            WHERE sex = ? AND year = ? AND {OBSERVED}
            ORDER BY total_count DESC
            LIMIT ?
            """,
            (sex, year, limit),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_name_history(name: str, sex: str) -> list[dict]:
    conn = database.connect()
    try:
        rows = conn.execute(
            f"""
            SELECT name, sex, year, total_count, popularity_percent, popularity_rank
            FROM names
            WHERE LOWER(name) = LOWER(?) AND sex = ? AND {OBSERVED}
            ORDER BY year
            """,
            (name, sex),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_latest_data_year() -> int | None:
    """The newest year with a recorded count, read from the data itself.

    Forecast eligibility is defined against this rather than a hardcoded year,
    so next year's data refresh needs no code change.
    """
    conn = database.connect()
    try:
        row = conn.execute(f"SELECT MAX(year) AS newest FROM names WHERE {OBSERVED}").fetchone()
        return row["newest"]
    finally:
        conn.close()
