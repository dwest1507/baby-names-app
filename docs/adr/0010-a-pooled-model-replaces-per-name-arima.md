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

**Training is bounded to the most recent 40 origins.** The pool used to start at 1930 and weight a
1935 name-origin exactly like a 2014 one. Research swept the window and found an interior optimum
at four decades — both less history and more of it cost skill — and found that discarding the older
rows outright beats discounting them with a half-life by about a factor of three on ranks 101–1000
(FINDINGS-6.md §2). So this is a bound on how much history the model wants, not a claim that the
1940s are a different process. It makes training cheaper as a side effect, not as its purpose.

**What the boosters produce is not yet what the site draws.** Three corrections are applied on top,
in this order, by `pooled.point_forecasts`:

1. **Growth cap.** Each path's implied `|log(forecast / origin share)|` is clipped at the 99.9th
   percentile of the five-year moves names actually made, read off the training outcomes rather
   than invented. A model working in log space can extrapolate without limit — research measured a
   five-year ratio of 2.5e44 on nine name-origins — and one such forecast is a visibly broken
   chart. It is a clip, not a shrink: an ordinary forecast passes through bit-for-bit.
2. **Path smoothing.** A three-point moving average over the path's log steps, padded at both
   edges. One booster per horizon means nothing ties the five together, so the line can rise, dip
   and rise again without that ever having been a claim about the name. The padding makes each raw
   step enter the smoothed sum exactly three times, so the smoothed steps sum to the raw ones and
   **the five-year endpoint is unchanged**: it redistributes years one through four and nothing
   else. Research measured it as an accuracy change rather than a cosmetic one — reversals roughly
   halve, and poolSkill rises +0.0024 / +0.0023 / +0.0012 in the top three tiers, costing 0.0009 in
   the deep tail (FINDINGS-6.md §3).
3. **Reconciliation to the corpus total.** `popularity_percent` is a share within a sex, so across
   every name of a sex in a year it sums to a fixed total; forecasting names one at a time lets the
   sum drift, and it drifts upward, which means predicting growth for more names than can grow. One
   multiplicative factor per (year, sex, horizon) scales each slice onto the total that sex actually
   held at the origin. Multiplicative and global is the measured choice: it is a constant shift in
   log space, so every ratio between two names survives it and nothing is pushed toward zero. The
   equal-absolute (`ols`) and volatility-weighted spreads were both measured and are not used, and
   there is no per-tier path — a tier is a property of the evaluation, not of the adding-up
   constraint.

**The order is load-bearing.** Smoothing preserves each path's endpoint but moves h1–h4, so it
changes the very sums reconciliation targets; reconciling first and smoothing afterwards would
break the adding-up again at four of five horizons. Research composed them in this order and
measured that they still pay together (FINDINGS-6.md §3, "Where it goes in the pipeline").

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

**Skill is measured across the whole backtest span, not on one window.** The batch fits at every
origin from 1995 whose five-year window has since closed — 26 of them on the 2025 database,
`range(1995, MAX(year) - 4)`, read off the data rather than written down — and scores every
eligible name at each. What a name carries is the average of its own window skills, beside the
count of windows behind it.

One window would not support the claim the search page makes. The most recent one is 2021–25,
which is largely a measurement of the birth-rate shock rather than of the name, and next year's
rebuild would relabel every name with a different shock. Averaging over 26 windows dilutes any one
period. Names eligible at fewer origins are averaged over fewer windows rather than excluded: a
name first recorded in 2010 is scored on the windows it has, because the alternative leaves most
of what visitors search with no figure at all.

**The span is one advancing window, not 27 independent fits.** Consecutive origins want training
sets that overlap in 39 of 40 years, and feature extraction — not boosting — is what this batch
spends its time on: rebuilding each fit's own rows is 1,080 origin-passes against 71.
`pooled.TrainingWindow` builds each origin once, hands it to every fit that wants it, and releases
it when the span has moved past. That is also what holds memory flat: what is retained is the
training window plus the handful of newer origins not yet in it, so backtesting 26 origins costs
what fitting one does. Measured on the real database: 53 s for the first origin, which builds the
whole window, then 12-18 s for each of the remaining 26 fits, at a peak of 1.5 GB.

**The artifact certifies itself.** `model_evaluation` holds one row per popularity tier —
`pool_skill`, `med_skill`, and the span (`origins_evaluated`, `min_origin`, `max_origin`) they were
measured over. It is in the database rather than in a build log because the deploy gate is handed
an artifact and nothing else, months after the run that produced it. Two scores rather than one,
because they fail differently: `pool_skill` sums absolute errors before dividing and so is
dominated by the names whose forecasts are most wrong, while `med_skill` is the median window
skill, and a model that is excellent on the giants and useless below them passes the first and
fails the second. A tier the span never populated is absent rather than zero, so a gate can tell
"measured, and bad" from "never measured". What reads it is tracked separately.

**Coverage stays where it was.** The bands are calibrated on one origin's residuals (ten years
back), so counting a 1995 outcome against them would measure a band built from a later era's
errors — an anachronism, not a larger sample. Interval coverage therefore remains the holdout
origin's, as ADR 0011 describes it, and only the point skill is averaged across the span.

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

- **The point stack is applied at every origin the batch fits, not only the production one.** The
  bands are calibrated on the residuals of capped, smoothed, reconciled forecasts and the holdout is
  scored against the same, so `calibration` and each name's `validation` describe the forecasts the
  search page actually draws.
- **The parity fixture pins the stack too, and pins the cap separately.** A bound at the 99.9th
  percentile of real moves is one no ordinary forecast reaches, so nothing in the fixture is
  clipped and the published numbers alone would not notice a cap derived from the wrong rows.

- **Measured on the real database** (2,181,032 observed rows, 1880-2025; 26 origins, 24,285
  forecasts, 17,449 of them scored), the whole batch takes **7.0 minutes** — less than the 8.8 the
  three-fit batch took, because the advancing window removes far more feature extraction than 24
  extra fits add. poolSkill and medSkill by tier, against the naive baseline:

  | tier | poolSkill | medSkill | ARIMA poolSkill (ADR 0010 context) |
  |---|---|---|---|
  | top100 | **0.374** | 0.437 | 0.161 |
  | top1000 | **0.238** | 0.215 | — |
  | top5000 | **0.096** | 0.071 | −0.211 |
  | rest | **0.077** | 0.059 | −0.413 |

  Positive in every tier, which is what the replacement was for, and above the 0.345 the research
  measured for the free forecast on the top 100. Interval coverage is unchanged at 0.783 / 0.939,
  which is the check that the advancing window did not quietly change the model.
- **Window counts are real, not nominal.** Of the 17,449 scored names, 7,474 carry all 26 windows
  and 9,975 carry fewer, down to one. The `skill_windows` figure beside the skill is what tells a
  visitor which they are looking at.
- **`validation.skill` no longer describes the holdout.** `mae`, `rmse`, `mape` and `points` still
  do — they are what the predicted-against-actual table shows — but `skill` beside them is the
  span-wide average, and `skill_windows` says how many windows it rests on. The search page says so
  rather than continuing to claim the figure is the holdout's.
- **The sample database gained a short-lived name.** Every profile in it was previously eligible for
  the entire span, so a batch that silently dropped names eligible at fewer origins would have
  passed. `Aria` is recorded from the mid-2000s and clears the ten-year minimum partway through,
  which is the shape most real names have.
- **`model_evaluation` is preserved across a `names` rebuild**, like `forecasts` and `calibration`,
  so reingesting the source cannot leave an artifact that still carries forecasts but can no longer
  say what they scored.

What is deliberately not done here, and is tracked separately: the visual demotion of the forecast
line, and the `verify-db` deploy gate that reads `model_evaluation`.
