"""Generate a small sample names.db for local development and tests.

The real database is ~1.1 GB and tracked with Git LFS. This script builds a
tiny stand-in with a handful of names and plausible multi-decade trends so the
API and frontend can be developed without the full dataset.

The sample mirrors the shape of the built database: observed rows only, and
the same indexes. A name/sex/year whose share falls below the suppression
threshold simply has no row, exactly as the source data has none for a
combination with fewer than five recorded births. The index DDL is shared with
`build_db.py` (`app.db_schema`) so the two artifacts cannot drift apart.

Usage: uv run python scripts/make_sample_db.py [output_path]
"""

import math
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app import db_schema  # noqa: E402

YEARS = range(1960, 2025)

# Below this share of births the source suppresses the count for privacy, so
# the real data has no row at all for that name/sex/year.
SUPPRESSION_THRESHOLD = 1e-5

# (name, sex, peak_year, peak_percent, spread)
PROFILES = [
    ("Emma", "F", 2015, 0.011, 20),
    ("Olivia", "F", 2020, 0.012, 18),
    ("Sophia", "F", 2012, 0.010, 15),
    ("Mary", "F", 1962, 0.014, 30),
    ("Liam", "M", 2020, 0.011, 15),
    ("Noah", "M", 2016, 0.010, 18),
    ("Oliver", "M", 2022, 0.009, 16),
    ("David", "M", 1965, 0.016, 35),
    ("Michael", "M", 1970, 0.020, 40),
    # Falls out of use: recorded early, absent from the final year, so it has
    # plenty of history but is not a candidate for a forecast.
    ("Debra", "F", 1962, 0.008, 12),
    # Recent arrival: present in the final year but with fewer than ten
    # recorded years, so it fails the minimum-history guard.
    ("Mateo", "M", 2024, 0.004, 3),
]


def build(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("DROP TABLE IF EXISTS names")
    conn.execute(db_schema.CREATE_TABLE)

    rows = []
    for year in YEARS:
        births = 1_800_000  # rough per-sex birth cohort
        year_rows: dict[str, list[tuple[str, str, int, int, float]]] = {"M": [], "F": []}
        for name, sex, peak, peak_pct, spread in PROFILES:
            pct = peak_pct * math.exp(-(((year - peak) / spread) ** 2))
            if pct < SUPPRESSION_THRESHOLD:
                # Suppressed at the source: no row exists for this combination.
                continue
            year_rows[sex].append((name, sex, int(pct * births), year, pct))
        for entries in year_rows.values():
            entries.sort(key=lambda r: (-r[2], r[0]))
            for rank, (name, s, count, y, pct) in enumerate(entries, start=1):
                rows.append((name, s, count, y, pct, rank))

    conn.executemany("INSERT INTO names VALUES (?, ?, ?, ?, ?, ?)", rows)
    db_schema.create_indexes(conn)
    conn.commit()
    conn.close()
    print(f"Wrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "data/sample_names.db")
