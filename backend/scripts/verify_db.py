"""Verify a built database artifact is complete and usable before deploying.

A deploy with a missing, truncated, or LFS-pointer database builds and starts
happily -- FastAPI never touches sqlite until the first request -- and then
cannot answer anything. This is the last check before an artifact is
published or a container is shipped: it confirms the file is a real SQLite
database (not an unresolved LFS pointer or unrelated garbage, reusing
`app.database`'s existing diagnosis so this reports the same distinctions the
backend itself would at startup) and that both tables the app depends on --
`names` and the precomputed `forecasts` (see
docs/adr/0004-forecasts-as-a-build-artifact.md) -- are present and non-empty.

Usage: uv run python scripts/verify_db.py [db_path]
Exits non-zero with a message on stderr if the artifact is not usable.
"""

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app import database  # noqa: E402

REPO_ROOT = Path(__file__).parent.parent.parent
DEFAULT_DB = str(REPO_ROOT / "data" / "names.built.db")

REQUIRED_TABLES = ("names", "forecasts")


class VerificationError(RuntimeError):
    """Raised when a database artifact fails verification."""


def verify(db_path: str) -> dict[str, int]:
    """Check db_path is a complete, usable database artifact.

    Returns a row count per required table on success. Raises
    VerificationError with a human-readable reason on failure.
    """
    problem = database.describe_db_problem(db_path)
    if problem is not None:
        raise VerificationError(problem)

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        try:
            tables = {
                row[0]
                for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
            }
        except sqlite3.DatabaseError as e:
            raise VerificationError(
                f"`{db_path}` could not be read as a SQLite database: {e}"
            ) from e

        missing = [table for table in REQUIRED_TABLES if table not in tables]
        if missing:
            raise VerificationError(
                f"`{db_path}` is missing table(s): {', '.join(missing)}. "
                "Was `make build-db` (and, for `forecasts`, "
                "`make precompute-forecasts`) run against this artifact?"
            )

        counts = {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in REQUIRED_TABLES
        }
    finally:
        conn.close()

    empty = [table for table, count in counts.items() if count == 0]
    if empty:
        raise VerificationError(
            f"`{db_path}` has table(s) with zero rows: {', '.join(empty)}. "
            "This looks like a truncated or incomplete build."
        )

    return counts


def main() -> None:
    db_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DB
    try:
        counts = verify(db_path)
    except VerificationError as e:
        print(f"FAILED: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"OK: {db_path}")
    for table, count in counts.items():
        print(f"  {table}: {count:,} rows")


if __name__ == "__main__":
    main()
