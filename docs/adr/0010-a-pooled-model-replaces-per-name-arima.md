# 10. A pooled model replaces per-name ARIMA

Date: 2026-09-12

## Status

Accepted

Supersedes [7. The precompute batch runs in parallel, on a smaller grid, resumably](0007-precompute-batch-runs-in-parallel.md).

Amends [4. Forecasts as a build artifact](0004-forecasts-as-a-build-artifact.md).

Builds on [9. Relational join index and reproducible SSA data ingestion](0009-relational-join-index-and-reproducible-ingestion.md).

## Context

`/search` fitted one ARIMA per name, from that name's own history and nothing else. Six rounds of
rolling-origin research (`research/forecasting/`, FINDINGS.md through FINDINGS-6.md) measured what
that bought, and for most names the answer was: less than nothing.

| popularity tier | ARIMA poolSkill | pooled model poolSkill |
| --- | --- | --- |
| top 100 | +0.161 | **+0.345** |
| 1001–5000 | **−0.211** | positive |
| beyond 5000 | **−0.413** | positive |

`poolSkill` is `1 - Σ|error| / Σ|naive error|` over a tier. Negative skill means the dashed line a
visitor read was *further* from what actually happened than a flat line would have been. The site
was shipping, for two of four tiers, a forecast that left a reader worse informed than ignoring the
chart.

The cause is structural rather than a tuning failure. A name has thirty to a hundred annual
observations, which is not enough to identify an ARIMA order; the grid search was mostly selecting
between random walks. ADR 0007 already noticed this from the other end — across a 60-name sample,
21 names selected ARIMA(0,1,0) and 16 selected (1,1,0) — and recorded it as a hint that "most names
carry no forecastable signal beyond a random walk". The hint was right about the *model* and wrong
about the *signal*: the signal is there, it is just not in one name's own thirty points. It is in
what names shaped like this one have gone on to do, which only a model trained across every name
can see.

Three further things came due at the same time:

1. **The batch's whole shape existed to survive the per-name fit.** ADR 0007 bought a 54x speedup
   with a worker pool, a per-name timeout enforced by killing the worker, and a resume flag —
   machinery whose entire purpose was to get 24,700 independent CPU-bound grid searches finished
   without one pathological series outlasting the rest.
2. **`statsmodels` and `scipy` are heavy.** ADR 0004 kept them out of the runtime image, but they
   remained build dependencies with their BLAS libraries behind them.
3. **The database moved to 2025 and gained `idx_names_name_sex_year`** (ADR 0009, PR #52). That
   left the artifact in a contradictory interim state: 2025 was simultaneously an observed year in
   `names` and the first forecast year in `forecasts`, which are legacy 2025–2029 ARIMA rows. On
   the search chart the forecast point overwrote the record.

## Decision

**Fit one model across every name, not one model per name.**

`backend/scripts/forecast/pooled.py` trains a LightGBM regressor per forecast horizon
(h ∈ {1..5}) over every eligible name-origin at once, and predicts every eligible name in a single
pass. The configuration is the plain one research settled on:

- **Target** `log(y[t+h] / y[t])` — growth relative to the origin, so a prediction is scale-free
  and a share is recovered by multiplying the origin year's share by its exponential.
- **Features** the base block only (13 columns: growth over 1/2/3/5/10 years, acceleration,
  volatility, level, level², distance below peak, years since peak, age, years since trough). No
  hand-built interactions, no cohort block, no lifecycle block — research measured each and the
  base block is what it selected.
- **Weights** `share^0.5`, clipped, normalised to mean 1: the fit leans toward the names visitors
  actually look up without letting a handful of giants become the whole fit.
- **One seed, one booster per horizon.** Horizons are predicted directly rather than recursively,
  so an error at h=1 cannot compound to h=5.

**Origins.** The batch trains at three, each strictly in the past relative to what it is used for:

| origin | what it produces |
| --- | --- |
| newest observed year (2025) | the served forecast, 2026–2030 |
| newest − 5 (2020) | each name's holdout validation, scored against the observed 2021–2025 |
| newest − 10 (2015) | the log-residual quantiles the published bands are built from |

The third is what keeps ADR 0005 honest. Bands calibrated on the holdout they are then measured
against would report a coverage of "about the nominal level" by construction. Calibrating a
window earlier makes the figure in the `calibration` table an out-of-sample measurement, which is
the only kind worth publishing. (The single global quantile per horizon used here is an interim
band; stratifying it by popularity tier and volatility is tracked separately.)

**Extraction streams from the index.** One query — `SELECT name, sex, year, popularity_percent,
popularity_rank FROM names ORDER BY name, sex, year` — is exactly the order
`idx_names_name_sex_year` (ADR 0009) already stores, so SQLite walks the index and returns rows
already grouped and already sorted. `pooled.stream_series` closes each series as its run ends, so
there is no temporary B-tree, no sort file, and no intermediate artifact on disk. The research
harness's `series.npz` has no production equivalent.

**The batch loses its scaffolding.** `--workers`, `--timeout` and `--resume` are removed. There is
one fit, so there is nothing to fan out, nothing that can run away, and nothing to resume. What
replaces them is `--threads`, pinned rather than dynamic, because LightGBM's histogram
construction is only bit-for-bit reproducible at a fixed thread count. `make precompute-forecasts`
and `make publish-db` are unchanged in interface.

**`model` becomes a global model card.** One model forecasts every name, so an ARIMA order, an
AIC, and four residual-diagnostic p-values are describing a fit that no longer happens. The
`model_card` table holds one row — model class, target, features, training window, seed — and the
API serves it under `model`. It is stored once rather than copied into 24,700 payloads, for the
same reason `calibration` is its own table.

**The port is pinned numerically.** `research/forecasting/make_parity_fixture.py` runs a generated
set of series through the *research* modules and writes both the series and the predictions those
modules produced to `backend/tests/fixtures/pooled_parity.json` (~380 KB, checked in).
`backend/tests/test_forecast_pooled.py` replays the same series through the shipped code and
demands the same features and the same forecasts to nine significant figures. The fixture is the
reason that check runs in CI without the 1.1 GB database, and the reason the backend test suite
imports nothing from `research/`.

## Consequences

Forecast quality is the point: positive skill in every tier, roughly double in the top 100. What
else moves:

- **2025 is history again.** The forecast covers 2026–2030, so the year-collision the interim
  artifact carried is gone. `TrendChart` also refuses a forecast point for a year the history
  already covers, because the database is published independently of the code (ADR 0006) and a
  deploy can meet a year-old artifact.
- **No value can be negative.** A forecast is a positive share times an exponential, and each band
  edge is that times another exponential. The `max(x, 0.0)` clamps that guarded against ARIMA's
  additive intervals reaching below zero are removed rather than left as dead defence.
- **The batch is minutes, not half an hour.** The sample-database suite drops from ~50 s to ~5 s.
  On the real database the cost is dominated by three fits over ~640k training rows rather than by
  24,700 grid searches.
- **Rebuilds are reproducible.** Pinned seed plus pinned threads means two runs over the same
  database produce identical forecasts, which is what makes a model change distinguishable from
  noise. `tests/test_forecast_pooled.py` checks this across two processes, on the parity fixture
  — the sample database is too small for LightGBM to make a single split, so determinism there is
  a statement about a constant.
- **`lightgbm` and `scikit-learn` join the dev/batch group.** They are absent from the runtime
  image for the same reason `statsmodels` is, and `tests/test_runtime_dependencies.py` blocks all
  four at import and runs the app for real.
- **`pebble` is no longer used by the batch.** It stays a declared dependency for now; removing it
  is a separate change.
- **ARIMA stays in the tree.** `scripts/forecast/arima.py` is no longer reachable from the batch,
  but `research/forecasting/methods.py` imports it as the frozen historical arm so rounds 1–6
  remain reproducible.

What is deliberately not done here, and is tracked separately: the 40-origin training window, path
smoothing and reconciliation to the corpus total; conformal bands stratified by tier × volatility;
per-name skill averaged across all rolling origins; the visual demotion of the forecast line; and
the `model_evaluation` table with the `verify-db` deploy gate that reads it.
