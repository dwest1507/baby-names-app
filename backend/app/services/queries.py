"""Parameterized queries against the names table.

The table holds observed rows only: a row exists for a name/sex/year only if a
count was actually recorded against it. Because the source suppresses counts
below five, a missing row means "fewer than five, or none" — never "zero". See
docs/adr/0003-observed-rows-only.md.
"""

import json
import sqlite3

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


# The whole-population row, stored alongside the per-stratum ones. Mirrors
# `scripts/forecast/pooled.GLOBAL_STRATUM`, which the request path cannot
# import: `pooled` lives outside the container image (ADR 0004).
GLOBAL_STRATUM = ("*", -1)


def get_calibration(tier: str | None, volatility_bin: int | None) -> dict[str, dict]:
    """Measured interval calibration for one name's stratum, keyed by level.

    `calibration` holds one row per `(nominal_level, tier, volatility_bin)`,
    written by scripts/precompute_forecasts.py from a holdout backtest across
    every eligible name. The row returned is the one for the stratum this
    name is served under, so the chart can say what the band covered for
    names *like this one* rather than for the average name — which is the
    whole reason the table is keyed this way. See
    docs/adr/0011-conformal-bands-keyed-by-strata.md.

    A name whose stratum was never populated by the backtest falls back to the
    whole-population row, which is also the only row a band it could have been
    given was built from. Each returned row names the stratum it describes, so
    a fallback is visible rather than silent.

    Empty when the table is absent or predates the strata keying — the
    database is published independently of this code (ADR 0006), so a deploy
    can meet an artifact keyed by nominal level alone, whose rows cannot be
    read as describing any stratum. That costs the band label, which the chart
    already handles, rather than the forecast.
    """
    conn = database.connect()
    try:
        rows = conn.execute(
            "SELECT nominal_level, tier, volatility_bin, empirical_coverage, n "
            "FROM calibration WHERE (tier = ? AND volatility_bin = ?) "
            "OR (tier = ? AND volatility_bin = ?) "
            "ORDER BY tier = ? ASC",
            (tier, volatility_bin, *GLOBAL_STRATUM, GLOBAL_STRATUM[0]),
        ).fetchall()
        calibration: dict[str, dict] = {}
        for row in rows:
            # The global rows sort last, so a stratum row already present is
            # never overwritten by the fallback.
            calibration.setdefault(
                str(row["nominal_level"]),
                {
                    "nominal": row["nominal_level"],
                    "tier": row["tier"],
                    "volatility_bin": row["volatility_bin"],
                    "empirical_coverage": row["empirical_coverage"],
                    "n": row["n"],
                },
            )
        return calibration
    except sqlite3.OperationalError:
        return {}
    finally:
        conn.close()


def get_model_card() -> dict | None:
    """What produced the forecasts: the model, its training set, its features.

    One pooled model forecasts every name (see
    docs/adr/0010-a-pooled-model-replaces-per-name-arima.md), so this is a
    single row rather than something stored per name.

    None when the batch has not run against this artifact — including when the
    table does not exist at all. The database is published independently of
    this code (ADR 0006), so a deploy can meet an artifact built before
    `model_card` existed; that should cost the model panel, not the whole
    forecast endpoint.
    """
    conn = database.connect()
    try:
        row = conn.execute("SELECT payload FROM model_card WHERE id = 1").fetchone()
        return json.loads(row["payload"]) if row else None
    except sqlite3.OperationalError:
        return None
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
        # The stratum columns are read defensively for the same reason
        # `get_model_card` tolerates a missing table: the database is
        # published independently of this code (ADR 0006), so a deploy can
        # meet an artifact from before those columns existed. A forecast
        # without a stratum is still a forecast.
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(forecasts)")}
        has_stratum = {"tier", "volatility_bin"} <= columns
        row = conn.execute(
            "SELECT payload"
            + (", tier, volatility_bin" if has_stratum else "")
            + " FROM forecasts WHERE name = LOWER(?) AND sex = ?",
            (name, sex),
        ).fetchone()
        if row is None:
            return None
        stored = json.loads(row["payload"])
        # The calibration stratum the batch served this name under, carried
        # alongside the payload so the endpoint can look up the coverage
        # measured for names like it without recomputing anything.
        stored["stratum"] = (row["tier"], row["volatility_bin"]) if has_stratum else (None, None)
        return stored
    finally:
        conn.close()
