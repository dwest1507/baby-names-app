"""Verify a built database artifact is complete and usable before deploying.

A deploy with a missing, truncated, or LFS-pointer database builds and starts
happily -- FastAPI never touches sqlite until the first request -- and then
cannot answer anything. This is the last check before an artifact is
published or a container is shipped: it confirms the file is a real SQLite
database (not an unresolved LFS pointer or unrelated garbage, reusing
`app.database`'s existing diagnosis so this reports the same distinctions the
backend itself would at startup) and that the tables the app depends on --
`names`, the precomputed `forecasts` (see
docs/adr/0004-forecasts-as-a-build-artifact.md) and `model_evaluation` -- are
present and non-empty.

It is also where the forecasting programme's acceptance rule is enforced,
rather than remembered. The batch measures each popularity tier's skill
against the naive "no change" baseline across the whole backtest span and
writes it into the artifact, so the gate can read what a build scored months
after the run that produced it (ADR 0010). A build ships if the top 100 clears
`MIN_TOP_TIER_SKILL`, every tier below it beats the baseline, and the span
behind those scores is the whole one this database supports -- a backtest that
stopped early scored something different from what it claims to have scored.

Usage: uv run python scripts/verify_db.py [db_path]
Exits non-zero with a message on stderr if the artifact is not usable.
"""

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app import database  # noqa: E402
from scripts.forecast import pooled  # noqa: E402

REPO_ROOT = Path(__file__).parent.parent.parent
DEFAULT_DB = str(REPO_ROOT / "data" / "names.built.db")

REQUIRED_TABLES = ("names", "forecasts", "model_evaluation")

# The floor the top 100 has to clear. ARIMA scored 0.161 there and the pooled
# model 0.374 (ADR 0010); 0.30 sits between them with room for the noise a
# rebuild on a new year of data moves the figure by.
MIN_TOP_TIER_SKILL = 0.30


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
                "Was `make build-db` (and, for `forecasts` and "
                "`model_evaluation`, `make precompute-forecasts`) run against "
                "this artifact?"
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

    _check_model_evaluation(db_path)

    return counts


def _read_model_evaluation(db_path: str) -> tuple[list[tuple], int]:
    """The scores the artifact certifies itself with, and its newest records."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT tier, pool_skill, origins_evaluated, min_origin, max_origin "
            "FROM model_evaluation"
        ).fetchall()
        (newest,) = conn.execute("SELECT MAX(year) FROM names").fetchone()
    finally:
        conn.close()
    order = {tier: i for i, tier in enumerate(pooled.TIERS)}
    return sorted(rows, key=lambda row: order.get(row[0], len(order))), newest


def _check_model_evaluation(db_path: str) -> None:
    """Enforce the acceptance rule on the scores the artifact carries."""
    rows, newest = _read_model_evaluation(db_path)

    scores = {tier: pool_skill for tier, pool_skill, _, _, _ in rows}

    # What the batch running to completion against *this* database would have
    # covered -- so next year's artifact is held to twenty-seven origins with
    # nothing here edited. See `pooled.backtest_span`.
    expected = pooled.backtest_span(newest)
    span = f"{expected.start}:{expected.stop - 1}"
    for tier, _, origins, first, last in rows:
        if (origins, first, last) != (len(expected), expected.start, expected.stop - 1):
            raise VerificationError(
                f"`{db_path}` scored {tier} over {origins} origin(s) "
                f"({first}:{last}), not the {len(expected)} ({span}) a complete "
                f"backtest against its own {newest} records covers. This looks "
                "like a truncated or partially-resumed run."
            )

    top_tier = "top100"
    if top_tier not in scores:
        raise VerificationError(
            f"`{db_path}` carries no {top_tier} score in `model_evaluation` "
            f"(it has: {', '.join(sorted(scores)) or 'nothing'}). A tier the "
            "backtest never populated is absent rather than zero, so this "
            "artifact cannot be certified either way."
        )
    if scores[top_tier] < MIN_TOP_TIER_SKILL:
        raise VerificationError(
            f"`{db_path}` scores {scores[top_tier]:.2f} pool_skill on {top_tier}, "
            f"below the {MIN_TOP_TIER_SKILL:.2f} a deployable build must clear "
            f"(short by {MIN_TOP_TIER_SKILL - scores[top_tier]:.2f})."
        )

    lower = {tier: skill for tier, skill in scores.items() if tier != top_tier}
    if not lower:
        raise VerificationError(
            f"`{db_path}` carries a {top_tier} score and nothing below the top 100. "
            "The acceptance rule is about the tail as much as the giants, and an "
            "artifact scored only on the giants has not been measured against it."
        )

    flat = {tier: skill for tier, skill in lower.items() if skill <= 0}
    if flat:
        failed = ", ".join(f"{tier} ({skill:+.2f})" for tier, skill in sorted(flat.items()))
        raise VerificationError(
            f"`{db_path}` scores no better than assuming no change on: {failed}. "
            "Every popularity tier below the top 100 must beat the naive "
            "baseline for a build to ship."
        )


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

    # What the gate just certified, so a maintainer reading the build log sees
    # the scores they are about to publish rather than only that they passed.
    rows, _ = _read_model_evaluation(db_path)
    for tier, pool_skill, origins, first, last in rows:
        print(f"  {tier}: pool_skill {pool_skill:.3f} over {origins} origins ({first}:{last})")


if __name__ == "__main__":
    main()
