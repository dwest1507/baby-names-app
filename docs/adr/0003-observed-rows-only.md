# 3. The database stores observed rows only

Date: 2026-08-31

## Status

Accepted

## Context

The data pipeline cross-joined every name/sex pair with every year in the data and zero-filled
the gaps, so a name absent from a year still carried a row with `total_count = 0`. This was done
deliberately, to give trend analysis a rectangular series to work with.

Measured against the real database, that decision cost more than it bought:

| | rows |
|---|---|
| Total | 16,899,750 |
| Fabricated by the cross-join | 14,750,273 |
| Actual observations | 2,149,477 |

87% of the rows — and of the 1.1 GB file — are invented. The table is an exact cross-product of
116,550 name/sex pairs and 145 years, so the padding is fully reconstructible and nothing is
lost by removing it.

The size alone blocks deployment: 1.1 GB is impractical to bake into a container image,
incompatible with downloading at startup in a container that sleeps, and a persistent volume
reintroduces the always-on cost that scale-to-zero exists to avoid.

The padding is also a modelling defect, not just a storage one. A padded series is dominated by
decades of zeros, so the ARIMA grid search picks mean-reverting models that pull forecasts back
toward an inflated long-run average — the app predicted *Karen* rising fivefold over five years
beside a holdout MAPE of 284%. Removing the padding raises measured interval coverage from 0.710
to 0.767 on its own, and un-blocks the log transform, whose `(series > 0).all()` guard was
denying it to 694 of the top 1000 names of 2024: precisely the modern, fast-growing names where
a multiplicative model matters most.

Separately, the history lookup filters on `LOWER(name)`, which no index on the bare `name` column
can serve. On the real database the planner fell back to `SEARCH names USING INDEX idx_sex
(sex=?)` plus a temporary B-tree for the sort — scanning one sex's 8.4 million rows, measured at
1436 ms per lookup. This was invisible in development because the sample database carried
adequate indexes the real one did not: a production-only defect by construction.

## Decision

**The database stores observed rows only.** A row exists for a name/sex/year only if a count was
actually recorded against it. The cross-join and zero-fill are removed from the pipeline
(`data_pipeline.ipynb`), and `backend/scripts/build_db.py` produces the deployable artifact from
the pipeline's output by keeping the rows with a recorded count.

**A missing row means "fewer than five, or none" — never "zero".** The SSA suppresses counts
below five for privacy, so every surviving row has `total_count >= 5` (verified: the minimum in
the built database is exactly 5). This distinction is load-bearing and is stated in the chatbot's
schema description, so it never asserts "0 babies" for a year whose count was merely suppressed.

**A composite index on `(LOWER(name), sex, year)`** replaces the previous single-column indexes,
and the history query is written to match that expression exactly.

**One definition of the index shape, in `backend/app/db_schema.py`,** used by both the build
script and the development sample generator, so a production-only index defect cannot recur. The
notebook repeats the DDL because it is a separate pipeline with its own `requirements.txt` and
cannot import backend code; the repetition is commented as such at both ends.

## Consequences

Measured on the real data:

| | before | after |
|---|---|---|
| Rows | 16,899,750 | 2,149,477 |
| File size | 1.1 GB | 137 MB |
| History lookup | 1436 ms (scan of 8.4M rows) | 0.98 ms (index seek) |
| Query plan | `SEARCH ... USING INDEX idx_sex` + temp B-tree | `SEARCH ... USING INDEX idx_names_lower_name_sex_year (<expr>=? AND sex=?)` |

- The file is now small enough to publish as a build artifact and bake into a container image.
- **Behaviour did not move.** The app was already filtering these rows out at the query layer
  (ADR 0001's slice), so deleting them was a storage change with the behaviour already proven.
  That query-layer filter is now removed as redundant. The existing API and query tests passed
  unchanged across both steps, which is the evidence.
- Any future consumer of this table must not read a missing row as a zero. Averages over "the
  years a name was recorded" and averages "per year since 1880" are different questions, and only
  the first one is answerable from this data.
- The history query's `LOWER(name) = LOWER(?)` predicate is now load-bearing text: rewriting it
  in a way that no longer matches the indexed expression silently reintroduces the scan. The
  query plan is asserted in `backend/tests/test_sample_db.py` so that regression fails a test
  rather than a production request.
