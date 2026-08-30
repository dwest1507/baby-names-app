"""Generate a small sample names.db for local development and tests.

The real database is ~1.1 GB and tracked with Git LFS. This script builds a
tiny stand-in with a handful of names and plausible multi-decade trends so the
API and frontend can be developed without the full dataset.

Usage: uv run python scripts/make_sample_db.py [output_path]
"""

import math
import sqlite3
import sys
from pathlib import Path

YEARS = range(1960, 2025)

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
]


def build(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("DROP TABLE IF EXISTS names")
    conn.execute(
        """
        CREATE TABLE names (
            name TEXT,
            sex TEXT,
            total_count INTEGER,
            year INTEGER,
            popularity_percent REAL,
            popularity_rank INTEGER
        )
        """
    )

    rows = []
    for year in YEARS:
        births = 1_800_000  # rough per-sex birth cohort
        year_rows = {"M": [], "F": []}
        for name, sex, peak, peak_pct, spread in PROFILES:
            pct = peak_pct * math.exp(-(((year - peak) / spread) ** 2))
            if pct < 1e-5:
                continue
            year_rows[sex].append((name, sex, int(pct * births), year, pct))
        for entries in year_rows.values():
            entries.sort(key=lambda r: r[2], reverse=True)
            for rank, (name, s, count, y, pct) in enumerate(entries, start=1):
                rows.append((name, s, count, y, pct, rank))

    conn.executemany("INSERT INTO names VALUES (?, ?, ?, ?, ?, ?)", rows)
    conn.execute("CREATE INDEX idx_names_name_sex ON names (name, sex)")
    conn.execute("CREATE INDEX idx_names_sex_year ON names (sex, year)")
    conn.commit()
    conn.close()
    print(f"Wrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "data/sample_names.db")
