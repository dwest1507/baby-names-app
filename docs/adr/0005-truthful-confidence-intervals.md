# 5. Truthful confidence intervals and skill against the naive baseline

Date: 2026-08-31

## Status

Accepted

## Context

Every forecast ships with 80% and 95% confidence bands and the chart shades them. ADR 0001
measured that removing the padding raised interval coverage from 0.710 to 0.767 for the 95%
band — real progress, but still short of 0.95. The design session's underlying study — the one
that found 71% coverage for a nominal 95% band across 300 interval checks — explicitly called
itself underpowered at 60 names. Continuing to print "95%" over a band whose true coverage is
unknown at scale is a calibration defect, not a sampling artefact, and it stays a defect
regardless of how close the number gets without being *measured*.

Separately, the validation panel reported MAE and RMSE in scientific notation
(`1.23e-5`) — unreadable on a page meant for a portfolio visitor — and a bare MAPE percentage
with nothing to compare it against, so a forecast performing far worse than assuming no change
looked like just another number next to one that beats it.

`forecast._validate` already refits on a 5-year holdout for every eligible name during the
precompute batch (ADR 0004) and scores MAE/RMSE/MAPE. That refit already produces exactly what's
needed to measure coverage honestly: the same training-only fit can also emit the 80%/95%
intervals it *would* have published, and checking whether the actual holdout value fell inside
them is a one-line comparison. Doing this for every eligible name — not a sample — is what turns
the existing holdout validation into the wide calibration benchmark the PRD wants: two orders of
magnitude more interval checks than the 60-name study that raised the question.

## Decision

**Bands are relabelled with the coverage they actually achieve, measured across every eligible
name's holdout backtest. Intervals are not algebraically widened.**

- `forecast._validate` now also computes, for each holdout point, whether it fell inside the
  80% and 95% intervals from the training-only fit (`coverage`), and a `skill` score against a
  naive baseline (below). Both ride along in the same refit that already produces MAE/RMSE/MAPE
  — no extra model fitting.
- `scripts/precompute_forecasts.py` aggregates `coverage` across every eligible name into hits
  and totals per nominal level, and writes one row per level into a new `calibration` table
  (`backend/app/db_schema.py`, `CREATE_CALIBRATION_TABLE`): `nominal_level`,
  `empirical_coverage`, `n`. The raw per-name `coverage` arrays are popped out of the stored
  `forecasts` payload before it's written — they exist only to be aggregated, and keeping them
  per-name would be dead weight in every response (consistent with ADR 0004's minimal-payload
  decision).
- `GET /api/names/{name}/forecast` reads `calibration` (`queries.get_calibration`) and includes
  it in the response — the same aggregate for every name, `null` only when there's no forecast
  to draw bands for. The frontend (`TrendChart.tsx`) labels the shaded areas with
  `empirical_coverage`, not the nominal level: `"51% interval"`, not `"95% interval"`.
- **Widening was considered and rejected for now.** Scaling interval half-widths to hit a target
  nominal coverage (conformal-style) is the other option the parent PRD allows, and it isn't
  ruled out for later. It was rejected here because it needs a stable coverage estimate to scale
  against, and the only measurement available before a real-database run is the sample-DB
  batch's 45 holdout points per level (9 eligible names × 5 holdout years — see Consequences).
  A scaling factor derived from 45 points would be at least as likely to be wrong as the label
  it replaces, and it would need to be silently re-derived once the real 24,721-name batch runs.
  Relabelling has no such dependency: it states whatever the latest batch measured, stays correct
  by construction as better measurements arrive, and needs no separate recalibration step.
  Simplicity is also an explicit tie-breaker in the PRD's own wording, and relabelling is the
  simpler of the two given what these numbers turned out to be.

**Skill is `1 - model_mae / naive_mae`**, where the naive baseline is the standard
"no change"/persistence forecast: the last training-observed value repeated for every holdout
year. 0 means the model is no better than assuming nothing changes; negative means it's worse.
It's computed once per name during the same batch refit and stored in `validation.skill`.

**A forecast that loses to naive (negative skill) is flagged, not suppressed.** The search page
shows it with a warning (`Notice variant="warning"`) rather than hiding the forecast outright,
because the history and the forecast shape are still informative even when the point forecast
underperforms a flat line — suppressing the chart entirely would throw away the history view
too. This is the simpler of the two options the PRD allows and matches how the page already
handles the "no forecast" cases (explaining, not just omitting).

**Validation figures render in fixed-decimal percentage points**, reusing the existing
`formatPercent` helper (already used elsewhere on this page) instead of `.toExponential(2)`:
`1.23e-5` becomes `0.0012%`.

## User-facing copy (drafted, not final)

Per the parent PRD's stated norm for this repo, this wording is expected to be reviewed rather
than merged unread:

- Positive skill: *"Beats the naive "no change" baseline by 25.0%: on the holdout years, this
  model's error was that much smaller than simply repeating the last recorded value."*
- Negative skill: *"This forecast performs worse than simply assuming no change — its holdout
  error was 10.0% higher than the naive baseline's. Treat the forecast and its confidence bands
  with caution."*
- Band legend labels: `"51% interval"` / `"44% interval"` (measured), replacing the previous
  `"95% interval"` / `"80% interval"` (nominal).

## Consequences

**Measured on the sample database** (`backend/scripts/make_sample_db.py`, 9 eligible names out
of 11 profiles — Debra and Mateo are deliberately ineligible, see ADR 0001):

| nominal level | empirical coverage | n (holdout points) |
|---|---|---|
| 80% | 0.444 | 45 |
| 95% | 0.511 | 45 |

These numbers are **not the production figures** and should not be read as a replacement for
the design session's 60-name study. The sample database has 9 eligible name/sex pairs, each
contributing exactly `VALIDATION_YEARS` (5) holdout points, so each level's coverage estimate
rests on 45 points from 9 synthetic, smoothly-peaked Gaussian trend profiles — a much smaller
and much easier-to-fit dataset than the real 24,721 eligible pairs. **Real-database calibration
numbers are still pending the full precompute batch run against `data/names.built.db`**, which
issue #9 measured at roughly 30 hours single-threaded and deliberately deferred; this issue does
not run it either, per its own constraints. The `calibration` table and the relabelling
mechanism are proven correct against the sample database; the trustworthy coverage numbers a
visitor should actually see come from running `make precompute-forecasts` against the real
artifact before deploying, which is a known pending step.

- **No interval is labelled with a coverage level it does not achieve**, by construction: the
  label is read from the same table the batch just measured, not written down separately.
- **The mechanism generalizes without code changes once the real batch runs.** Re-running
  `scripts/precompute_forecasts.py` against the real database overwrites the `calibration`
  table with numbers from all 24,721 eligible pairs, and the frontend label updates
  automatically — no redeploy of frontend code is needed to reflect newly-measured coverage.
- **A visitor may see an oddly low percentage** (e.g. "44% interval") on a sample deployment
  seeded from a small dataset. This is intentional: a truthful low number is the entire point of
  this ADR, and it's expected to read differently — likely more favourably — once measured
  against the real, much larger dataset.
- `validation.skill` and `calibration` are additive payload fields; existing consumers of the
  unchanged fields (`mae`, `rmse`, `mape`, `points`) are unaffected. `Validation` and
  `ForecastPayload` in `frontend/lib/api.ts` were updated to match (payload shape changes are
  explicitly permitted in this slice of the PRD, unlike ADR 0004's).
- The `calibration` table has no `IF NOT EXISTS`-guarded consumer until the batch runs at least
  once — `queries.get_calibration()` will raise `sqlite3.OperationalError` against a database
  that predates this change and hasn't been rebuilt. This is the same deploy discipline ADR 0004
  already established for the `forecasts` table: running the batch after every data refresh is a
  required step, not an optional optimization.

## Related

- Parent PRD: dwest1507/baby-names-app#5
- Depends on: ADR 0001 (forecast eligibility, the `_validate` holdout this reuses), ADR 0004
  (the precompute batch this extends, and the minimal-payload principle `coverage` follows by
  not being stored per-name)
- Implements: dwest1507/baby-names-app#10
