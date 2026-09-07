# 9. Relational join index and reproducible SSA data ingestion

Date: 2026-09-07

## Status

Accepted

## Context

The Groq-powered chatbot generates SQL queries based on natural language questions. For cross-year comparisons and rising-name queries (e.g. comparing 2024 against 2023), the model generates self-joins on `names` using `JOIN ... USING (name, sex)` or `ON a.name = b.name AND a.sex = b.sex`.

The database previously carried only two indexes:
1. `idx_names_lower_name_sex_year ON names (LOWER(name), sex, year)` (an expression index serving `/api/history`)
2. `idx_names_sex_year ON names (sex, year)` (serving top-name rankings)

Because `idx_names_lower_name_sex_year` is an expression index on `LOWER(name)`, SQLite cannot use it for equality joins on the bare `name` column. The query planner fell back to `idx_names_sex_year`, producing an unindexed nested loop join comparing 31,904 rows from 2024 against ~16,000 rows from 2023. This required over 510 million comparisons, taking over 60–90 seconds and exceeding the 5-second `QUERY_BUDGET_SECONDS` established in ADR 0008.

Furthermore, the original data ingestion workflow relied on an ad-hoc Jupyter notebook (`data_pipeline.ipynb`) that produced a zero-padded intermediate database (`data/names.db`) not tracked in Git, preventing reproducible end-to-end database builds from SSA source data.

## Decision

1. **Add `idx_names_name_sex_year`**:
   Add `CREATE INDEX idx_names_name_sex_year ON names (name, sex, year)` to canonical indexes in `backend/app/db_schema.py`. This allows cross-year self-joins on `(name, sex)` to execute via index seeks in ~0.3 seconds instead of >60 seconds.
2. **Nudge model toward CTE joins**:
   Update `SCHEMA_CONTEXT` in `backend/app/services/chatbot.py` with explicit guidance for cross-year comparisons using CTEs and `USING (name, sex)`.
3. **Reproducible SSA Ingestion Script**:
   Replace the manual `data_pipeline.ipynb` with an automated ingestion pipeline in `backend/scripts/build_db.py` that downloads the SSA zip (using Selenium to navigate SSA's anti-bot protections, or using a local file if present), calculates observed popularity ranks and percentages, and populates `data/names.built.db` with canonical schema and indexes.
4. **Dev dependency isolation**:
   Place `pandas` and `selenium` into `[dependency-groups] dev` in `backend/pyproject.toml` so the production container image remains lean.
5. **Decouple Ingestion from Forecasts**:
   Keep `make build-db` focused on table creation and indexing (~15 seconds), leaving `make precompute-forecasts` as a separate, resumable multi-core batch step.

## Consequences

- Relational joins across years on `name` and `sex` run in sub-second time, well within the 5.0-second budget.
- The deployable database can be cleanly reproduced once a year when new SSA data is published.
- Existing precomputed forecast models are preserved during in-place index additions.
- Unit tests in `test_sample_db.py` and `test_chatbot_sql.py` verify that year-over-year join query plans never regress to full scans.
