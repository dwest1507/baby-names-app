"""The shape of the names database: its table, and the indexes it must carry.

This lives in one place because the app is served from two differently-built
artifacts — the real database built by ``scripts/build_db.py`` and the sample
built by ``scripts/make_sample_db.py`` — and they must not drift apart.

They did drift, and it cost a production-only defect: the sample carried a
usable index for the history lookup and the real database did not, so a query
that scanned millions of rows in production planned perfectly in development
and in CI. Both builders now take their index DDL from here.
"""

CREATE_TABLE = """
CREATE TABLE names (
    name TEXT,
    sex TEXT,
    total_count INTEGER,
    year INTEGER,
    popularity_percent REAL,
    popularity_rank INTEGER
)
"""

# The history lookup is `WHERE LOWER(name) = LOWER(?) AND sex = ? ORDER BY year`.
# An index on the bare `name` column cannot serve it — `LOWER(name)` is not
# sargable — so the planner falls back to whatever else it can find, which on
# the real database meant scanning one sex's 8.4 million rows. Indexing the
# expression itself is what makes the lookup a seek, and carrying `sex` and
# `year` in the same index means the equality and the ordering both come out of
# it, with no temporary B-tree for the sort.
INDEXES = (
    "CREATE INDEX idx_names_lower_name_sex_year ON names (LOWER(name), sex, year)",
    # Serves the top-names query, which is by sex and year.
    "CREATE INDEX idx_names_sex_year ON names (sex, year)",
)


def create_indexes(conn) -> None:
    for statement in INDEXES:
        conn.execute(statement)


# Precomputed forecasts, keyed on the lowercased name and sex — the same key
# the history lookup normalizes to. `payload` is the JSON-encoded forecast
# blob (forecast points, model diagnostics, validation) with the history
# series stripped out: history is composed at request time from `names`
# instead, so this table never re-adds the size pruning just removed. See
# docs/adr/0004-forecasts-as-a-build-artifact.md.
# `coverage_hits`/`coverage_n` are this name's own contribution to the
# `calibration` aggregate: how many of its holdout points fell inside the
# interval the training-only fit would have published, per nominal level, as a
# JSON object keyed by level. They live here rather than in `payload` because
# `payload` is served to the API verbatim and coverage is a population
# statistic, not a per-name one.
#
# Storing them at all is what lets `calibration` be computed as a SUM over this
# table instead of from an accumulator local to one batch invocation. Without
# that, a resumed batch would calibrate on only the names it happened to refit
# — a non-random slice, since rows are written in name order — and publish a
# coverage figure that no longer describes the data. See
# docs/adr/0007-precompute-batch-runs-in-parallel.md.
CREATE_FORECASTS_TABLE = """
CREATE TABLE IF NOT EXISTS forecasts (
    name TEXT NOT NULL,
    sex TEXT NOT NULL,
    payload TEXT NOT NULL,
    coverage_hits TEXT,
    coverage_n TEXT,
    PRIMARY KEY (name, sex)
)
"""

# One row per nominal interval level (0.8, 0.95), holding the coverage that
# level actually achieved across every eligible name's holdout backtest — not
# a sample. `empirical_coverage` is the fraction of holdout points that fell
# inside the interval a training-only fit would have published; `n` is the
# number of holdout points behind that fraction (eligible names x
# VALIDATION_YEARS). The app must never label a band with `nominal_level` if
# `empirical_coverage` says otherwise. See
# docs/adr/0005-truthful-confidence-intervals.md.
CREATE_CALIBRATION_TABLE = """
CREATE TABLE IF NOT EXISTS calibration (
    nominal_level REAL NOT NULL PRIMARY KEY,
    empirical_coverage REAL NOT NULL,
    n INTEGER NOT NULL
)
"""
