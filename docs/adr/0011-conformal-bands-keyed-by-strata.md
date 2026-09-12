# 11. Conformal bands, keyed by popularity tier and volatility bin

Date: 2026-09-12

## Status

Accepted

Supersedes [5. Truthful confidence intervals and skill against the naive baseline](0005-truthful-confidence-intervals.md).

Builds on [10. A pooled model replaces per-name ARIMA](0010-a-pooled-model-replaces-per-name-arima.md).

## Context

ADR 0005 caught a real defect and did the honest thing about it: the 95% band a visitor read
covered the outcome 51% of the time, so the chart was relabelled to say "51% interval". That was
truthful, and it left two things unfixed.

**The band was still not calibrated.** ADR 0005 explicitly deferred widening or recalibrating,
and gave a good reason: the only measurement then available was 45 holdout points from the sample
database — 9 synthetic names × 5 years — and a scaling factor derived from 45 points is at least
as likely to be wrong as the label it replaces. That constraint is gone. ADR 0010's pooled model
forecasts every eligible name from one fit, so a holdout at one origin scores all ~24,700 of them
at once, which is ~123,000 interval checks rather than 45.

**One number described every name.** `calibration` held one row per nominal level, so the figure
on the chart was a population average. A population average is exactly what conceals the failure
it is supposed to expose. Research measured the shape of that concealment directly
(`research/forecasting/intervals.py`, FINDINGS-5 §4): splitting the 80% holdout by how far a name
sat below its own peak, a band conditioned on nothing swung 5.8 points of coverage across the bins
and 6.4 points in its low-side miss rate. A band can have textbook marginal coverage and still be
wrong for every name in it — under-covering the names at their peak by falling short on the
downside, over-covering the ones long past it — and the visitor who is being misled is the one
reading the name the model is least sure about.

Two axes explain most of that. **Popularity tier** is the axis every number in this programme has
been reported on since round 1, because the tiers behave differently enough that a pooled figure
is not informative about any of them. **Volatility** — the standard deviation of a name's recent
log steps, which is already a feature the model reads — is what separates two names at the same
rank whose futures are not equally knowable. Research conditioned on `tier × volatility` in round
3 (`conformal.py`) and that construction, `resid_tiervol`, has been the incumbent band since.

Round 5 tested replacing it with directly fitted quantile boosters. The verdict was a tier split,
not a sweep: `direct` wins ranks 1–100 decisively (interval score 0.696 against 0.735 at 80%,
P=100%) and loses ranks 101–1000 and the deep tail at P=100%, at a cost of 30 boosters per origin
instead of 5 and roughly ten minutes per origin instead of thirty-five seconds. FINDINGS-5 §8 says
plainly that a tier-switched band is a post-hoc split and should be confirmed on a fresh block
before it ships. It has not been. So this ADR ships the incumbent, and leaves the quantile
boosters where round 5 left them.

## Decision

**Bands are empirical quantiles of the pooled model's own five-year log residuals, taken within
`(popularity tier, volatility bin)`, and the coverage the API reports is the coverage measured for
that same cell.**

**Construction.** The batch fits at three origins (ADR 0010). At the earliest — `newest − 10` —
the five-year outcome has long since been observed, so the residuals
`log(actual) − log(published forecast)` are real. Those residuals are grouped by stratum, and each
group's two-sided quantiles at 80% and 95% become that stratum's band offsets
(`pooled.strata_bands`). They are applied multiplicatively, `forecast × exp(offset)`, so both
edges stay positive and no clamp is needed. The residuals are taken against the *finished* point
stack — capped, smoothed and reconciled (ADR 0010's amendment) — not the boosters' raw output, so
the band describes the errors of the line the chart actually draws.

**Strata.** `TIERS = (top100, top1000, top5000, rest)` by rank; `VOLATILITY_BINS = 3` by tertiles
of `vol`, the same feature the model reads. The tertile edges are cut from the calibration
origin's own rows rather than hardcoded — "volatile" is only meaningful relative to the corpus,
and a fixed cut leaves a bin empty on one database and holding everything on another — and are
then held fixed across the other two origins, so bin 2 means the same thing whether a name is
being calibrated, scored, or served.

**A stratum needs `MIN_STRATUM_ROWS = 60` observed outcomes to get a band of its own.** Below
that it uses the whole population's, because a tail estimated from six residuals is a confidently
wrong band rather than a conditioned one. Research's threshold, unchanged.

**Rank is read at the origin the row belongs to.** `stream_series` now carries a rank per year
rather than one per name, and `build_rows` records the rank the name held at its own origin. This
is what makes a tier a property of an origin year rather than of a name (CONTEXT.md, "Popularity
Tier"): a backtest at 1995 is tiered by 1995 ranks, so `top100`'s measured coverage describes the
names that were top-100 names then. Serving falls out of the same rule — the production origin
*is* the newest observed year, so a served name's tier is its rank in 2025. A null rank is an
absence of recorded popularity, so `popularity_tier` sorts it to `rest`; the bare `rank <= 100`
comparison it replaces filed it under `top100`, which is the one tier #47's deploy gate checks.

**`calibration` is keyed by `(nominal_level, tier, volatility_bin)`**, holding
`empirical_coverage` and `n` (`db_schema.CREATE_CALIBRATION_TABLE`). Each eligible name's holdout
points are counted into the cell it was in *at the holdout origin*, and `forecasts.tier` /
`forecasts.volatility_bin` record the cell each name is *served* under. A whole-population row is
stored per level under `tier = '*', volatility_bin = -1`; it is the fallback for a name whose own
stratum the backtest never populated, and the number worth quoting in release notes, not the
figure the endpoint prefers.

**The API returns the row matching this name.** `queries.get_calibration(tier, volatility_bin)`
selects the stratum's row, falling back to the population row, and every returned row names the
stratum it describes so a fallback is visible rather than silent. `TrendChart` labels each shaded
area with `empirical_coverage` as it already did — the change is in which coverage it is handed.

**Both halves of ADR 0005 that were right are kept.** Skill remains `1 - model_mae / naive_mae`
against the persistence baseline, a forecast that loses to naive is still flagged rather than
suppressed, and validation figures still render in fixed-decimal percentage points.

## Consequences

**Measured against `data/names.built.db`** (2,181,032 observed rows, 1880–2025; bands calibrated
at origin 2015, coverage measured on the 2021–2025 holdout from origin 2020; 87,245 interval
checks per level). Nine of the twelve cells cleared `MIN_STRATUM_ROWS` and got a band of their
own; the other three used the population's and their holdouts are the `*` row:

| nominal | stratum | coverage | n | | nominal | stratum | coverage | n |
|---|---|---|---|---|---|---|---|---|
| 80% | top100 · 0 | 0.813 | 980 | | 95% | top100 · 0 | 0.934 | 980 |
| 80% | top1000 · 0 | 0.773 | 8,470 | | 95% | top1000 · 0 | 0.924 | 8,470 |
| 80% | top1000 · 1 | **0.647** | 320 | | 95% | top1000 · 1 | 0.897 | 320 |
| 80% | top5000 · 0 | 0.796 | 22,185 | | 95% | top5000 · 0 | 0.950 | 22,185 |
| 80% | top5000 · 1 | 0.778 | 10,820 | | 95% | top5000 · 1 | 0.939 | 10,820 |
| 80% | top5000 · 2 | 0.776 | 5,270 | | 95% | top5000 · 2 | 0.927 | 5,270 |
| 80% | rest · 0 | 0.771 | 5,130 | | 95% | rest · 0 | 0.933 | 5,130 |
| 80% | rest · 1 | 0.782 | 16,105 | | 95% | rest · 1 | 0.937 | 16,105 |
| 80% | rest · 2 | 0.782 | 17,755 | | 95% | rest · 2 | 0.942 | 17,755 |
| 80% | `*` (fallback) | 0.776 | 210 | | 95% | `*` (fallback) | 0.919 | 210 |

Pooled over every cell: **0.783 at a nominal 80% and 0.939 at a nominal 95%**, against ADR 0005's
measured 0.444 and 0.511. That figure is printed by the CLI and deliberately not stored, because
it describes no single band.

Median published 95% band width, as the ratio of the upper edge to the lower at the five-year
horizon, by served stratum:

| tier | bin 0 (steadiest) | bin 1 | bin 2 (jumpiest) |
|---|---|---|---|
| top100 | **2.0** | — | — |
| top1000 | 3.3 | 5.5 | 6.2 |
| top5000 | 5.9 | 7.1 | 9.1 |
| rest | 6.1 | 6.0 | 6.7 |

A single population band would have given all of these the same width. A steady top-100 name now
gets one roughly four and a half times tighter than a jumpy tail name's, and both are honest.

- **No band is labelled with a coverage it does not achieve for names like the one being viewed.**
  Which is a stronger claim than ADR 0005's, and it is the one that matters: the previous
  guarantee was compatible with a band that was right on average and wrong in the tail.
- **A volatile name gets a visibly wider band than a steady name at the same rank.** That is the
  point of conditioning, and it is pinned as a property rather than a number
  (`tests/test_forecast_bands.py`).
- **The measurement is out-of-sample.** The bands are built at `newest − 10` and their coverage is
  measured at `newest − 5`, so the published figure is not a quantile scoring itself. ADR 0010
  added the third origin for exactly this reason.
- **A badly covered cell is now visible instead of averaged away.** `top1000 · 1` covers 0.647
  against a nominal 0.80 — 64 names, 320 checks. Under one population figure those names were
  shown 0.783. This ADR does not fix that cell; it makes it reportable, which is the
  precondition for fixing it, and #46 is what puts it in front of a visitor.
- **The issue's ~125,000 estimate is 87,245 in fact.** It assumed every one of the ~24,700
  eligible names has a complete five-year holdout. 24,451 are eligible at origin 2020 but only
  17,449 are observed in all of 2021–2025 — the rest fall below the source's suppression
  threshold somewhere in the window, and a name with a missing year has no outcome to check. The
  ~125,000 figure arrives with #45's 26-origin backtest, which multiplies this by the number of
  origins rather than by five.
- **Coverage is no longer comparable to ADR 0005's numbers**, because ADR 0005's numbers came from
  a parametric ARIMA interval on nine synthetic names and these come from empirical quantiles of a
  different model on the real corpus. There is no regression to read between them.
- **An artifact published before this change cannot carry its `calibration` rows forward.** They
  are keyed by level alone, so there is no stratum to file them under and no honest way to guess
  one. `build_db.py` drops them and `precompute_forecasts.py` rebuilds the table from `forecasts`,
  where the per-name coverage it aggregates actually lives. `forecasts` itself is widened in place
  and its rows survive.
- **The request path still fits nothing.** The stratum is stored on the forecast row, so serving a
  band is two indexed lookups, and the volatility tertiles — a property of the batch's calibration
  origin — never have to be recomputed to answer a request. ADR 0004 holds.
- **Directly fitted quantiles remain open.** Round 5 measured them as better on the top 100 and
  worse below it; when that split is confirmed on a fresh block, the band construction is the one
  place that changes, because everything downstream reads `calibration` by stratum either way.

## Related

- Parent PRD: dwest1507/baby-names-app#40
- Implements: dwest1507/baby-names-app#44
- Supersedes: ADR 0005 (the relabelling this replaces with a recalibration)
- Depends on: ADR 0010 (the pooled fit, its three origins, and the finished point stack the
  residuals are taken against), ADR 0001 (forecast eligibility), ADR 0004 (bands are a build
  artifact; nothing is fitted on the request path)
- Research: `research/forecasting/conformal.py`, `intervals.py`, FINDINGS-3 §8, FINDINGS-5 §§2–5
