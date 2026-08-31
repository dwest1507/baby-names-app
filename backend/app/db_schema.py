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
CREATE_FORECASTS_TABLE = """
CREATE TABLE IF NOT EXISTS forecasts (
    name TEXT NOT NULL,
    sex TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (name, sex)
)
"""
