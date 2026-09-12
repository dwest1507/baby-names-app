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
    # Serves relational equality joins on name and sex across different years.
    "CREATE INDEX idx_names_name_sex_year ON names (name, sex, year)",
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
# `tier`/`volatility_bin` are the calibration stratum this name is *served*
# under — read from its rank and its wobble in the newest observed year — and
# are what the API joins to `calibration` to report the coverage measured for
# names like this one. They are stored rather than derived at request time
# because the volatility bin edges are a property of the batch's calibration
# origin, and ADR 0004 keeps the request path free of anything computed.
# The columns `forecasts` carries, in the order the batch and `build_db.py`
# read and write them. Named here for the same reason as
# `CALIBRATION_COLUMNS`: a widened table must not leave a copy elsewhere
# reading a column list that no longer matches.
FORECASTS_COLUMNS = (
    "name",
    "sex",
    "payload",
    "coverage_hits",
    "coverage_n",
    "tier",
    "volatility_bin",
)

CREATE_FORECASTS_TABLE = """
CREATE TABLE IF NOT EXISTS forecasts (
    name TEXT NOT NULL,
    sex TEXT NOT NULL,
    payload TEXT NOT NULL,
    coverage_hits TEXT,
    coverage_n TEXT,
    tier TEXT,
    volatility_bin INTEGER,
    PRIMARY KEY (name, sex)
)
"""

# One row per (nominal level, popularity tier, volatility bin), holding the
# coverage that level actually achieved for names in that stratum across every
# eligible name's holdout backtest — not a sample. `empirical_coverage` is the
# fraction of holdout points that fell inside the interval a training-only fit
# would have published; `n` is the number of holdout points behind that
# fraction.
#
# It is keyed by the stratum rather than by the level alone because a single
# population figure conceals exactly the failure it exists to expose: bands
# can cover 80% of all names while covering 95% of the steady ones and 60% of
# the jumpy ones, and a visitor reading a jumpy name is shown the 80%. The
# stratum a row describes is the one measured at the *historical* origin, and
# `forecasts.tier`/`forecasts.volatility_bin` say which stratum each served
# name is in. `tier = '*'`, `volatility_bin = -1` is the whole-population row,
# kept for names whose own stratum was never measured.
#
# The app must never label a band with `nominal_level` if `empirical_coverage`
# says otherwise. See docs/adr/0011-conformal-bands-keyed-by-strata.md, which
# supersedes docs/adr/0005-truthful-confidence-intervals.md.
CREATE_CALIBRATION_TABLE = """
CREATE TABLE IF NOT EXISTS calibration (
    nominal_level REAL NOT NULL,
    tier TEXT NOT NULL,
    volatility_bin INTEGER NOT NULL,
    empirical_coverage REAL NOT NULL,
    n INTEGER NOT NULL,
    PRIMARY KEY (nominal_level, tier, volatility_bin)
)
"""

# The columns `calibration` carries, in the order the batch and `build_db.py`
# read and write them. Named here so a schema change cannot leave one of them
# copying a column list that no longer matches.
CALIBRATION_COLUMNS = (
    "nominal_level",
    "tier",
    "volatility_bin",
    "empirical_coverage",
    "n",
)


# The one thing that is true of the forecast model rather than of any one name:
# what it is, what it was trained on, and which features it reads. One pooled
# model produces every forecast (see
# docs/adr/0010-a-pooled-model-replaces-per-name-arima.md), so this is stored
# once here rather than copied into every row of `forecasts` — the same reason
# `calibration` is its own table. The single row is pinned by a CHECK so a
# second batch cannot quietly leave two cards behind for the API to pick
# between.
CREATE_MODEL_CARD_TABLE = """
CREATE TABLE IF NOT EXISTS model_card (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    payload TEXT NOT NULL
)
"""
