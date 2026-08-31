"""Build the deployable names database from the pipeline's output.

The pipeline's database cross-joins every name/sex pair with every year and
zero-fills the gaps, so most of its rows are fabricated rather than observed.
This script produces the artifact the backend is actually served from: the
observed rows only, carrying the indexes the app's queries need.

The result is what gets published and baked into the container image, so it is
a first-class command rather than a one-off — re-run it whenever the pipeline
produces a new database.

Usage: uv run python scripts/build_db.py [source_path] [output_path]
"""

import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app import db_schema  # noqa: E402

REPO_ROOT = Path(__file__).parent.parent.parent
DEFAULT_SOURCE = str(REPO_ROOT / "data" / "names.db")
DEFAULT_OUTPUT = str(REPO_ROOT / "data" / "names.built.db")


def build(source: str, output: str) -> dict:
    """Write the pruned, indexed database and return what it contains."""
    source_path, output_path = Path(source), Path(output)
    if not source_path.exists():
        raise SystemExit(f"Source database not found: {source_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    started = time.monotonic()
    conn = sqlite3.connect(str(output_path))
    try:
        conn.execute("PRAGMA journal_mode = OFF")
        conn.execute("PRAGMA synchronous = OFF")
        conn.execute(db_schema.CREATE_TABLE)
        conn.execute("ATTACH DATABASE ? AS src", (str(source_path),))

        (source_rows,) = conn.execute("SELECT COUNT(*) FROM src.names").fetchone()
        # The only pruning rule: keep a row if a count was recorded against it.
        conn.execute(
            """
            INSERT INTO names (name, sex, total_count, year, popularity_percent, popularity_rank)
            SELECT name, sex, total_count, year, popularity_percent, popularity_rank
            FROM src.names
            WHERE total_count > 0
            """
        )
        conn.commit()
        conn.execute("DETACH DATABASE src")

        db_schema.create_indexes(conn)
        conn.execute("ANALYZE")
        conn.commit()
        (kept,) = conn.execute("SELECT COUNT(*) FROM names").fetchone()
    finally:
        conn.close()

    # VACUUM in its own connection so it is not inside the transaction above.
    conn = sqlite3.connect(str(output_path))
    try:
        conn.execute("VACUUM")
    finally:
        conn.close()

    return {
        "source_rows": source_rows,
        "rows": kept,
        "pruned": source_rows - kept,
        "bytes": output_path.stat().st_size,
        "seconds": time.monotonic() - started,
    }


def main() -> None:
    source = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SOURCE
    output = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT
    result = build(source, output)
    print(f"Source rows:  {result['source_rows']:,}")
    print(f"Pruned:       {result['pruned']:,} fabricated rows removed")
    print(f"Kept:         {result['rows']:,} observed rows")
    print(f"Wrote:        {output} ({result['bytes'] / 1024 / 1024:.1f} MB)")
    print(f"Took:         {result['seconds']:.1f}s")


if __name__ == "__main__":
    main()
