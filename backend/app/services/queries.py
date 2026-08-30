"""Parameterized queries against the names table."""

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
