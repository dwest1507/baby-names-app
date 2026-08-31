"""Precompute forecasts for every eligible name/sex and store them.

Forecasting is CPU-bound (up to 48 ARIMA fits, twice, per name) and the source
data only refreshes once a year, so fitting on every request buys nothing.
This script does it once, offline, and writes the result into the
`forecasts` table so the API becomes a lookup. See
docs/adr/0004-forecasts-as-a-build-artifact.md.

It reads the whole `names` table once and groups it in memory by (lowercased
name, sex) rather than querying per name — with ~24,700 eligible pairs in the
real database, one query per name would pay the lookup cost that many times
over. Fitting itself is delegated to `app.services.forecast.fit_forecast`,
the same code the API route used to call at request time, so stored values
cannot drift from what the code would produce live.

Runnable against either the sample database (fast, a handful of names) or the
real built database (~24,700 eligible pairs — this is the expensive case,
and is expected to take a long time; it is a batch/offline job, not something
run in a hot loop).

Usage: uv run python scripts/precompute_forecasts.py [db_path]
"""

import json
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app import db_schema  # noqa: E402
from app.services import forecast  # noqa: E402

REPO_ROOT = Path(__file__).parent.parent.parent
DEFAULT_DB = str(REPO_ROOT / "data" / "names.built.db")


def run(db_path: str) -> dict:
    """Fit and store a forecast for every eligible name/sex pair in db_path."""
    started = time.monotonic()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute(db_schema.CREATE_FORECASTS_TABLE)

        (latest_year,) = conn.execute("SELECT MAX(year) FROM names").fetchone()

        # The whole table, read once, then grouped in memory — not one query
        # per name/sex pair.
        rows = conn.execute(
            "SELECT name, sex, year, total_count, popularity_percent, popularity_rank "
            "FROM names ORDER BY name, sex, year"
        ).fetchall()

        groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in rows:
            groups[(row["name"].lower(), row["sex"])].append(dict(row))

        conn.execute("DELETE FROM forecasts")
        eligible = 0
        for (lowered_name, sex), history in groups.items():
            years = [r["year"] for r in history]
            if not forecast.is_eligible(years, latest_year):
                continue
            eligible += 1
            stored = forecast.fit_forecast(history)
            conn.execute(
                "INSERT INTO forecasts (name, sex, payload) VALUES (?, ?, ?)",
                (lowered_name, sex, json.dumps(stored)),
            )
        conn.commit()
    finally:
        conn.close()

    return {
        "groups": len(groups),
        "eligible": eligible,
        "seconds": time.monotonic() - started,
    }


def main() -> None:
    db_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DB
    result = run(db_path)
    print(f"Name/sex pairs:       {result['groups']:,}")
    print(f"Eligible & forecast:  {result['eligible']:,}")
    print(f"Took:                 {result['seconds']:.1f}s")


if __name__ == "__main__":
    main()
