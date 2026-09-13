# 12. A track record, indexed by horizon, replaces the holdout on the search page

Date: 2026-09-12

## Status

Proposed

Builds on [10. A pooled model replaces per-name ARIMA](0010-a-pooled-model-replaces-per-name-arima.md)
and [11. Conformal bands, keyed by popularity tier and volatility bin](0011-conformal-bands-keyed-by-strata.md).

## Context

The search page's evidence that a forecast is worth anything was the **Validation Holdout**: five
predicted-against-actual rows from a single origin, at horizons 1 through 5. That is a defensible
statistic and a poor argument. A visitor who does not read statistics is being asked to judge a
five-year forecast from five rows, each of which was made a different distance ahead — the 2021 row
one year out, the 2025 row five — with nothing on the page saying so. The rows are not comparable to
each other, and there are not enough of them to be comparable to anything else.

Meanwhile the batch already fits **26 backtest origins** (every origin from 1995 whose five-year
window has closed, at 2025) and predicts every eligible name at every one of them. It then throws
all of it away: `precompute_forecasts.py` does `del rows, predicted` at the end of each origin,
keeping only the summed skill in `pooled.BacktestTally` and the holdout origin's five points. The
most persuasive thing the batch knows — *here is what this model said about this name, twenty-six
times, and here is what happened each time* — is computed and discarded.

Two facts make the naive fix wrong. First, a year can be predicted from several origins at several
horizons, so "what the model said about 2023" is not one number, and a column that silently mixes
horizons means nothing down its length. Second, a *fixed* horizon is not right either: whether a
forecast is any good is a different question at one year out than at five, and which question a
visitor is asking is not knowable from here.

## Decision

**The page reports a horizon-indexed track record, and the holdout stops being displayed.**

**The track record is indexed by horizon.** For horizon `h`, year `Y`'s entry is the prediction
made at origin `Y − h`, paired with what was actually observed in `Y`. Each horizon is a complete
series in its own right; no entry mixes horizons with its neighbours.

**The visitor picks the horizon, and the page defaults to one year.** Every figure that depends on
it moves together: the projected share column, the projected rank column, the error column, and the
summary error figures above them. A default of one year is the question most visitors are actually
asking, and the honest one to answer first — a model looks better at `h=1` than at `h=5` and the
page should not open on its most flattering slice by accident, but it should open on the slice a
reader can interpret.

**Summary accuracy figures are derived from the rows on screen, not stored.** Whatever the visitor
can see in the Error column is exactly what the summary averages. This is not an optimisation: it
is the property that makes the summary checkable by hand, and it is unavailable to any design that
stores per-horizon summaries separately from the series they describe.

**The holdout remains in the batch and leaves the page.** It still sets and measures the conformal
bands (ADR 0011) — that job is unaffected and unchanged. What is removed is `validation.points`
from the served payload and the "Holdout validation" panel from the page. Keeping both would put
two different answers to "what did the model say about 2023" on one page — one from origin 2020 at
`h=3`, one from origin 2022 at `h=1` — with no visible reason for the difference.

**The batch fits origins the backtest span excludes, for the horizons they can answer.** The span
holds only origins whose *five-year* window has closed — 1995 through 2020 at 2025 — so a track
record built from it alone would end at 2021 for `h=1`, leaving the default view silent on the four
most recent years. Origins 2021–2024 are fitted as well and contribute the horizons whose outcomes
are observed, so every horizon runs to the newest year. **They contribute to the track record only.**
They are not added to `pooled.BacktestTally`, `validation.skill` or `model_evaluation`, all of which
are defined on closed five-year windows and whose figures the `verify_db.py` deploy gate reads; a
one-year window entering that arithmetic would move the gate's numbers without anyone deciding to.

**The track record is stored compactly and expanded by the API.** On disk each horizon is a start
year plus parallel arrays of shares and ranks; `forecast.py` expands them into per-year objects in
the response. Per-year JSON objects on disk would cost roughly 159 MB against roughly 59 MB for the
arrays, on an artifact that is downloaded from Hugging Face on every deploy (ADR 0006). Gzip already
removes the repeated keys on the wire, so the verbose form buys nothing there and costs 100 MB here.
This is the one place `payload` stops being served byte-for-byte verbatim; expanding an array is
composition, not fitting, so ADR 0004 holds.

**Stored floats are rounded to six significant figures.** The payload stores
`0.0020805187517257528` for a figure the page renders as `0.0021%`. `formatPercent(f, 4)` needs six
decimal places of the fraction, so six significant figures is lossless at any share magnitude and
roughly halves what every existing point costs.

## Considered options

**Keep the holdout and add the track record beside it.** Rejected: the two measure different things
and nothing on the page distinguishes them, so the disclosed MAE would not reconcile with the Error
column above it. On a page whose purpose is earning trust, a summary that fails a reader's
arithmetic check is worse than no summary.

**A fixed five-year horizon.** The original decision, reversed before implementation. It gives one
consistent column and answers only one of the questions a visitor has.

**Store per-horizon summary statistics alongside the series.** Rejected as redundant and
falsifiable: two sources for one number, which will eventually disagree.

## Consequences

- **The batch must retain what it currently discards.** Each backtest origin's predictions for all
  five horizons are accumulated rather than freed. At ~24,700 eligible names this is bounded and
  small in memory; the cost lands on the artifact, not the run.
- **The payload grows.** With the extra origins the record holds 140 entries per name — 30 at
  `h=1` down to 26 at `h=5` — each carrying a projected share and a projected rank, against a
  current payload of ~1,435 bytes. Compactly encoded and rounded this is roughly 2.4 KB per name,
  about 59 MB across ~24,700 names.
- **The batch grows by four fits**, roughly a minute on a run that takes about seven.
- **Two origin sets now exist and mean different things.** The scored span (closed five-year
  windows, feeding skill and the deploy gate) and the fitted set (span plus 2021–2024, feeding the
  track record). Code that conflates them silently changes what `verify_db.py` certifies.
- **`validation.points` disappears from the API contract**, and the frontend tests that pin the
  holdout table (`frontend/__tests__/search.test.tsx`) move with it.
- **"Validation Holdout" narrows in CONTEXT.md** to what it still does: measure band calibration.
  Its old definition claimed it existed for display on the search page, which this makes false.
- **A name eligible at few origins has a short track record**, and at some horizons none at all.
  The page must read an empty series as "not measured" rather than as a zero, exactly as
  `skill_windows` already qualifies `skill`.

## Related

- Parent PRD: dwest1507/baby-names-app#59
- Depends on: ADR 0010 (the pooled fit and its rolling backtest span), ADR 0004 (forecasts are a
  build artifact; the request path fits nothing), ADR 0011 (the holdout's surviving job)
- Introduces: CONTEXT.md "Track Record"
- Paired with: ADR 0013 (the projected rank each track record entry carries)
