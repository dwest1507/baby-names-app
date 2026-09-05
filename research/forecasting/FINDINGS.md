# Improving the popularity forecast

Research run: 2026-09-05, against `data/names.built.db` (2,149,477 observed rows, 1880–2024,
21,792 name/sex series in current use). Every number below was measured; none is quoted from
literature. Reproduce with the harness in [README.md](README.md).

Popular names were treated as the thing to optimise, so results are broken down by 2024 rank
tier (1–100, 101–1000, 1001–5000, >5000) and the headline metric — `poolSkill` — sums absolute
errors, which lets big names dominate it the way they dominate page views.

---

## 1. Where the shipped model actually stands

From the `forecasts` table itself (the full precompute: 21,750 names with a holdout
validation, 2020–2024):

| 2024 rank | names | median skill | mean skill | worse than "no change" | median MAPE |
|---|---|---|---|---|---|
| 1–100 | 200 | 0.276 | −0.233 | **33.5%** | 9.4% |
| 101–1000 | 1,781 | 0.030 | −0.312 | **37.3%** | 15.5% |
| 1001–5000 | 7,308 | 0.000 | −0.430 | 45.6% | 26.7% |
| >5000 | 12,461 | −0.032 | −0.892 | 54.1% | 43.2% |

Two things to read out of it. The model does add value on the most-visited names (median skill
0.276 in the top 100) — but **a third of top-100 names and 37% of the rest of the top 1000 are
forecast worse than a flat line**, and mean skill is negative in every tier, meaning the
failures are not small. The `Notice variant="warning"` for negative skill (ADR 0005) is not an
edge case on this page; it fires for roughly one popular name in three.

### Intervals are miscalibrated in exactly the wrong direction

The published `calibration` row says 0.859 coverage for the 80% band and 0.934 for the 95% one.
That aggregate is carried by the long tail. Recomputed per tier from `forecasts.coverage_hits`:

| 2024 rank | 80% band covers | 95% band covers | holdout points |
|---|---|---|---|
| 1–100 | **0.570** | **0.708** | 1,000 |
| 101–1000 | 0.672 | 0.782 | 8,905 |
| 1001–5000 | 0.832 | 0.916 | 36,540 |
| >5000 | 0.906 | 0.969 | 62,305 |
| *published aggregate* | *0.859* | *0.934* | *108,750* |

A visitor looking up Olivia sees a band labelled "93% interval" that covers 71% of the time for
names like hers. ADR 0005 chose to relabel bands with measured coverage rather than widen them;
the measurement it relabels with is a population average that describes almost nobody, and it is
most wrong for the names people search for. This is a live, user-visible defect independent of
any modelling change below.

---

## 2. Three concrete bugs in `forecast.py`

**a. The log transform is a no-op.** `_preprocess` applies `np.log1p` when the coefficient of
variation exceeds 0.5. The series it is applied to are *shares*: median 1.5e-5, 99th percentile
5.1e-3, maximum 4.6e-2. Over the whole dataset the largest gap between `log1p(x)` and `x` is
**2.25%** — `log1p` is the identity here. The gate fires for 12,816 of 21,792 series, so the
code believes it is modelling multiplicatively 59% of the time while fitting an additive model
100% of the time. Consequences: forecasts can go negative and get floored (**8.5% of stored
forecasts end at exactly 0**), and intervals are symmetric on the share scale, which is why the
bands are too narrow on big names and too wide on small ones. The fix is `np.log` on a positive
series, which is what ADR 0001 believed it had un-blocked.

**b. Second differencing on the names that matter.** Across all names the modal order is
(0,1,0), but among top-1000 names six of the eight most common orders carry `d=2`
((0,2,2), (0,2,1), (2,2,0), (0,2,0), (2,2,2), (2,2,1)). `d=2` extrapolates a *linear trend* five
years out. Capping differencing at 1 — one line in `_find_optimal_differencing`'s caller — is
worth more than any other single change to the existing pipeline:

| ranks 101–1000 | h1 | h2 | h3 | h4 | h5 |
|---|---|---|---|---|---|
| current | 0.105 | 0.154 | 0.139 | 0.101 | 0.060 |
| current, d ≤ 1 | 0.134 | 0.173 | 0.173 | 0.161 | **0.142** |

(skill vs naive by horizon; the shipped model's advantage decays to almost nothing by year five,
which is the year the chart draws most prominently.)

**c. The time index is wrong for 80.5% of series.** Rows are observed-only (ADR 0003), so a
name with a suppressed year has a gap — and the ARIMA is fit on the values as if they were
consecutive. 17,532 of 21,792 series have at least one gap; so do 1,232 of the 1,981 top-1000
series (their gaps are in the sparse pre-war years). Only 2.1% of top-1000 names have a gap in
the last 20 years, so fitting on a recent window — rather than reindexing 145 years — is the
cheap fix, and it also drops the pre-1950 regime that the AICc search currently gets to vote on.
*Windowed fitting was not benchmarked here; it is the most promising untested item.*

---

## 3. What beats it

Rolling-origin backtest, two origins (fit ≤2014 → score 2015–2019; fit ≤2019 → score 2020–2024),
every 2024 top-1000 name plus 1,200 mid and 1,200 tail names. n is name-origins.

**Ranks 1–100 (n=398)**

| method | poolSkill | medSkill | %>naive | med MAPE |
|---|---|---|---|---|
| `combo_t1k_current` — pooled(top-1k) ⊕ current | **0.332** | **0.370** | 71.9% | **7.7%** |
| `combo_t1k_ens` — pooled(top-1k) ⊕ ensemble | 0.324 | 0.337 | **74.1%** | 8.0% |
| `combo_pooled_ens` | 0.291 | 0.252 | 72.6% | 8.4% |
| `current_s70` — current, shrunk 70% toward last value | 0.274 | 0.323 | 71.1% | 8.3% |
| `pooled_t1k` — global ridge trained on top-1000 | 0.272 | 0.318 | 72.1% | 8.6% |
| `ens_all4` | 0.271 | 0.294 | 71.1% | 8.2% |
| `arima_d1` | 0.259 | 0.176 | 67.1% | 8.4% |
| **`current` (shipped)** | 0.213 | 0.315 | 66.3% | 8.7% |
| `pooled_ridge` — global ridge trained on all names | 0.090 | 0.043 | 54.0% | 11.5% |
| `naive` | 0.000 | 0.000 | 0.0% | 13.1% |

**Ranks 101–1000 (n=3,510)**

| method | poolSkill | medSkill | %>naive | med MAPE |
|---|---|---|---|---|
| `combo_pooled_ens` | **0.180** | 0.129 | 67.9% | 13.3% |
| `combo_t1k_pooled_ens` | 0.173 | **0.183** | **70.3%** | **12.8%** |
| `pooled_ridge` | 0.159 | 0.097 | 62.4% | 14.0% |
| `ens_all4` | 0.125 | 0.120 | 67.4% | 13.4% |
| `pooled_t1k` | 0.096 | 0.210 | 63.4% | 13.5% |
| **`current` (shipped)** | 0.038 | 0.032 | 53.4% | 14.4% |
| `naive` | 0.000 | 0.000 | 0.0% | 16.5% |

**Ranks 1001–5000 (n=2,275) and >5000 (n=1,521)** — nothing wins by much, and the shipped model
loses badly (poolSkill −0.179 and −0.363; it is beaten by a flat line on 68% and 73% of names).
The best available answers are heavily shrunk: `pooled_s70` (0.038) and `combo_pooled_ens_naive`
(0.033) in the 1001–5000 tier; in the tail nothing clears naive by a meaningful margin.

### The three ingredients

1. **Damping / shrinkage toward the last value.** Undamped trend extrapolation is the single
   biggest source of the blow-ups, and how hard you damp dominates which model you damp: the
   same log-linear trend fit scores 0.033 poolSkill at damping 0.70 and −0.326 at 0.90 in the
   101–1000 tier. Shrinking the *shipped* ARIMA 70% toward the last observed value takes it from
   0.213 to 0.274 in the top 100 with no refitting at all.
2. **A pooled ("global") model.** One ridge per horizon, fit on log-growth across every name and
   every origin whose targets were already observable (549,662 name-origins for the 2014 origin,
   636,420 for 2019), using only per-series features: recent growth at several lags,
   acceleration, volatility, level, distance below the all-time peak, years since peak, age. It
   costs milliseconds for all 24,000 names and beats the shipped ARIMA fourfold in the
   101–1000 tier. Its learned coefficients are readable and match the fashion-cycle literature:
   `level` is negative and grows with horizon (−0.033 at h1 to −0.135 at h5 — the more popular a
   name is, the more it reverts), `g10` positive (long-run momentum persists), `g1`/`g2`/`accel`
   negative (recent sharp moves partly reverse).
3. **Combination, not selection.** Log-space averaging of a pooled model with an ARIMA/damped
   arm wins in every popular tier. Choosing *per name* is worse than not choosing: see §4.

### Tier matters, and it is not a rounding error

The pooled model trained on all names is mediocre on the top 100 (0.090) while the same model
trained on top-1000 names only is three times better there (0.272) — and vice versa in the tail,
where `pooled_t1k` is the worst thing tested (−0.234). Popular names and rare names follow
different dynamics, and one global fit splits the difference badly. Whatever ships should either
be fit per tier or given tier interactions.

### Illustration (fit ≤2019, actual 2020–2024)

| name | actual 2024 | naive | current | `combo_pooled_ens` | `combo_t1k_current` |
|---|---|---|---|---|---|
| Olivia (F) | 0.912% | 1.107% (MAPE 9.8%) | 1.183% (14.7%) | 0.993% (**4.4%**) | 1.058% (6.9%) |
| Emma (F) | 0.836% | 1.027% (14.8%) | 0.890% (4.7%) | 0.843% (**1.5%**) | 0.844% (1.9%) |
| Liam (M) | 1.292% | 1.150% (4.2%) | 1.384% (7.1%) | 1.087% (7.0%) | 1.208% (**2.4%**) |
| Mateo (M) | 0.659% | 0.503% (14.2%) | 0.753% (10.2%) | 0.624% (**4.6%**) | 0.710% (6.9%) |
| Luna (F) | 0.442% | 0.465% (**7.4%**) | 0.800% (35.4%) | 0.634% (16.2%) | 0.759% (31.1%) |

Luna is the honest counter-example: she peaked in 2022 and fell, and every trend-aware method
rode the rise off a cliff. Turning points remain unforecastable from one series alone — which is
the argument for damping and for wide, correctly-calibrated bands rather than for a cleverer
point forecast.

---

## 4. Per-name model selection does not work (measured)

The obvious idea — let each name pick its own model on its own holdout — was tested by choosing
each name's method on the 2014 origin and scoring that choice on 2019 (ranks 1–1000, n=1,939):

| strategy | poolSkill | medSkill | %>naive |
|---|---|---|---|
| fixed `combo_pooled_ens` for everyone | **0.227** | **0.133** | **67.0%** |
| per-name selection on the earlier holdout | 0.133 | 0.082 | 55.9% |
| oracle (best method in hindsight) | 0.531 | 0.381 | 87.7% |

Selection is *worse than not selecting*: five holdout points cannot identify which model suits a
name, so the choice is noise. The oracle gap (0.53 vs 0.23) says the signal is real and large —
but it has to be captured by features inside a pooled model, not by picking a winner per name.

---

## 5. Intervals: calibrate empirically, condition on tier and volatility

Split-conformal intervals — take the log-scale residual quantiles from the 2014 origin, apply
them to the 2019 origin's forecasts — for `combo_pooled_ens`:

| quantile scheme | nominal 80% | nominal 95% | 95% on ranks ≤1000 | median band width (×) |
|---|---|---|---|---|
| one global quantile | 0.796 | 0.946 | 0.989 | 6.6 |
| by popularity tier | 0.791 | 0.949 | 0.959 | 11.8 |
| by series volatility | 0.795 | 0.949 | 0.970 | 4.1 |
| **tier × volatility** | 0.780 | 0.946 | 0.955 | **3.5** |

Every scheme lands within a couple of points of nominal — compared with 0.708 actual coverage
for the shipped 95% band on top-100 names. Conditioning on tier × volatility gets there with
bands **3.3× narrower** than tier alone, because a stable popular name genuinely deserves a
tighter band than a jumpy rare one. Per-horizon half-widths (log scale) come out at 0.27 → 0.60
for 80% and 0.53 → 1.41 for 95% across h1 → h5, so the band still widens with horizon, as it
should.

This also retires the awkward UI copy from ADR 0005: bands can be labelled "80%" and "95%"
truthfully again, instead of "51% interval".

---

## 6. What I would do, in order

1. **Fix the two bugs** (`log1p` → `log` on a positive series; cap `d` at 1). Small diffs inside
   `forecast.py`, no new dependencies, no schema change. Removes the 8.5% of forecasts that end
   at exactly zero and roughly doubles five-year skill on popular names.
2. **Shrink the point forecast toward the last observed value** (`w≈0.7` in log space). Three
   lines, no refitting, +0.06 poolSkill on the top 100.
3. **Report skill and coverage per popularity tier**, not as one global number — the single
   `calibration` row is the misleading part, not the model. Same table, one extra column.
4. **Add the pooled ridge** as a second arm and average in log space. ~150 lines, milliseconds
   to fit for the whole database, and it is where most of the remaining gain is. Fit it per tier
   (or with tier interactions).
5. **Replace parametric intervals with conformal ones** conditioned on tier × volatility. The
   precompute already produces exactly the residuals this needs — `_validate` computes them
   today and throws them away after counting coverage.
6. **Do not build per-name model selection.** Measured worse than a fixed ensemble (§4).

Compute is not a constraint on any of this: the pooled model and every damped/shrunk variant are
microseconds per name, against ~0.5s for one ARIMA grid search. An ARIMA-free stack — pooled
ridge averaged with the damped-trend family (`combo_pooled_dampfam`) — scores 0.150 poolSkill on
ranks 101–1000 against the shipped 0.038, which means **the 30-hour precompute batch could become
a sub-minute job that also forecasts four times better**. Keeping ARIMA in the ensemble buys
about 0.03 poolSkill in that tier (0.180) and 0.07 in the top 100 (0.332 vs 0.263); that is the
trade to decide, and it is a trade about batch runtime, not about request latency.

## 7. Worth trying next

- **Windowed fits** (last 20–30 years). Fixes the gap/time-index defect for popular names and
  drops a pre-war regime from model selection. Untested here.
- **Cross-name features.** Name popularity moves in correlated clusters (phonetic neighbours,
  sibling names, gender crossovers); the pooled model currently sees each series alone. This is
  the most likely source of the remaining oracle gap.
- **Quantile regression** for asymmetric bands — a rising name's downside and upside are not
  symmetric even in log space.
- **More origins.** Two is enough to rank methods, not to size the gains precisely; the 2019
  origin also spans the 2020–21 birth-rate shock.

## 8. Caveats

- Two origins only (2014, 2019). Conformal calibration therefore rests on a single earlier
  origin, and the 2019 test window contains a genuine demographic shock.
- The evaluation sample is every top-1000 name plus 1,200 mid-tier and 1,200 tail names; tail
  numbers are noisier than popular-tier numbers by construction.
- 299 of 4,381 sampled series are missing from the per-series backtest: the run was stopped
  while a handful of names were still inside a runaway ARIMA fit (the same pathology
  `precompute_forecasts.py --timeout` exists to bound). 12 of the 1,981 top-1000 series are
  affected. If anything this flatters the ARIMA arms.
- Ridge penalty (λ=30) was not tuned; the pooled numbers are a floor, not a ceiling.
- `poolSkill` is popularity-weighted by construction and is sensitive to a few large errors;
  `medSkill` and `%>naive` are reported alongside it for exactly that reason. Where the three
  disagree, the method has a fat tail.
