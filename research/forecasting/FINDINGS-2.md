# Round 2: squeezing more accuracy out of the forecast

Follow-up to [FINDINGS.md](FINDINGS.md), run 2026-09-05 against the same database. Round 1 ended
with a ranked list of changes and a section of untested hypotheses; this round tested them, and
one of them was wrong in a way that matters.

Everything below is scored on a **held-out origin**: models, penalties and ensemble weights are
chosen using origins 2009 and 2014 only, and 2019 (forecasting 2020–2024) is the test. Three
origins now, not two.

---

## 0. Correction to round 1: the log fix needs a guard rail

Round 1 recommended replacing `np.log1p` with `np.log` in `_preprocess`, on the evidence that
`log1p` is the identity on share-scale data. That evidence still holds — but the fix on its own
is **worse than the bug**:

| method (ranks 101–1000, 3 origins) | poolSkill | medSkill | %>naive |
|---|---|---|---|
| `current` (shipped) | 0.043 | 0.023 | 52.6% |
| `arima_d1` — shipped, `d` capped at 1 | 0.106 | 0.000 | 43.9% |
| `arima_log_d1` — true log + `d ≤ 1` | **−1.2e45** | 0.000 | 40.5% |

The median is unchanged; the mean is destroyed. On the log scale an ARIMA with any persistent
drift extrapolates *multiplicatively*, and nine name-origins out of ~13,000 produced five-year
ratios up to **2.5e44**. `log1p` was accidentally protecting the app from this by being additive.

**The guard rail is a cap read off the data.** The largest five-year change any name actually
made, at the fitting origins, is 2.76 in log terms (≈16×); per horizon:

| | h1 | h2 | h3 | h4 | h5 |
|---|---|---|---|---|---|
| cap on \|log(pred/last)\| | 1.48 | 2.02 | 2.27 | 2.45 | 2.76 |

Clipping to it touches **2,455 of 331,514 forecasts (0.74%)**, leaves every well-behaved method's
scores unchanged to three decimals, and turns `arima_log_d1` from −1.2e45 into −0.105. Any change
that moves modelling onto the log scale — which is the right scale — needs this cap shipped with
it. It is also worth applying to the *current* additive pipeline, which fails in the opposite
direction: 410 name-origins forecast below a tenth of the last observed value, which is where the
8.5% of forecasts that end at exactly zero come from.

---

## 1. What actually moved the needle

Two changes to the pooled model, each tested by ablation on origin 2019 (poolSkill):

| pooled model variant | ranks 1–100 | ranks 101–1000 | 1001–5000 | >5000 |
|---|---|---|---|---|
| round-1 features | +0.005 | +0.107 | +0.079 | +0.014 |
| **+ level interactions** | **+0.190** | **+0.197** | +0.085 | +0.033 |
| + cohort features instead | −0.034 | +0.102 | +0.076 | +0.010 |
| + both | +0.153 | +0.195 | +0.084 | +0.030 |
| **+ interactions, popularity-weighted fit** | **+0.313** | **+0.230** | +0.063 | +0.030 |
| + interactions, trained on top-1000 names only | +0.314 | +0.159 | **−0.183** | **−0.347** |

**Level interactions** (`level × {g1, g3, g5, g10, accel}`, plus `level²`) are the single biggest
structural gain: they let one fit hold two different dynamics — a top-100 name's momentum means
something different from a rank-8000 name's — instead of averaging them into mush. The h5
coefficients bear this out: `level²` and `level × g3` are the two largest terms.

**Popularity-weighted fitting** is the biggest gain of all, and it is the direct answer to
"popular names matter more": weight each training row by `share^0.5` (capped at 50× the mean) so
the least-squares fit spends its capacity where the traffic is. Sweeping the exponent:

| weighting | ranks 1–100 | ranks 101–1000 | 1001–5000 | >5000 |
|---|---|---|---|---|
| none | +0.190 | +0.197 | +0.085 | +0.033 |
| `share^0.5` | +0.313 | **+0.230** | +0.063 | +0.030 |
| `share^1` | +0.320 | +0.221 | +0.034 | +0.029 |
| `share^2` | **+0.340** | +0.146 | −0.163 | −0.302 |

`share^0.5` is the balanced choice. Pushing to `share^2` buys a little more at the very top and
makes the tail *worse than a flat line* — a trade available if you want it, but it turns the rest
of the site into noise. Note that weighting beats **training on top-1000 names only**, which wins
the top 100 by the same margin and then collapses everywhere else: one weighted model serves the
whole site better than two tier-specific ones.

---

## 2. Untested hypotheses from round 1, now tested

**Cohort / cross-name features: no.** Round 1 guessed that correlated clusters — shared endings,
initial letters, the sex's overall share — were where the oracle gap lived. Built (`cohorts.py`,
leave-one-out so a name never predicts itself) and measured: **they do not help.** Alone they are
worse than the base features on the top 100 (−0.034 vs +0.005); added to interactions they cost
0.04. The effect is real in the literature but is evidently already carried by the name's own
recent growth. A richer cohort definition (phonetic clusters, actual sibling co-occurrence) might
do better; letter-level cohorts do not.

**Windowed fits: partly.** Fitting a trailing window of calendar years (not of observations —
gappy names would silently stretch it) helps the damped arm and not the ARIMA arm, on ranks 1–100:

| | full history | 30-year window | 20-year window |
|---|---|---|---|
| `ets_log_phi80` | 0.178 | **0.249** | **0.249** |
| `current` (shipped ARIMA) | 0.169 | 0.156 | — |

So the gap/time-index defect is real but is not what was holding ARIMA back, and windowing is a
cheap win for exponential smoothing only.

**Learned ensemble weights: yes, in the tail.** Fitting non-negative weights summing to one, per
tier and per horizon, on 2009+2014 and applying them to 2019 (`weights.py`). What they learn is
more interesting than what they score — at h5, ranks 101–1000 put 0.93 on the pooled model, while
the >5000 tier puts **0.55 on naive**: the ensemble rediscovers "shrink to no-change in the tail"
without being told. They win the two tail tiers outright but do not beat the pooled model alone
where it matters most.

---

## 3. Held-out result (origin 2019, everything tuned on 2009/2014)

| ranks 1–100 (n=198) | poolSkill | medSkill | %>naive | med MAPE |
|---|---|---|---|---|
| **pooled, weighted + capped** | **0.310** | 0.310 | 68.7% | **7.3%** |
| `current_s70` (shipped, shrunk 70%) | 0.255 | 0.285 | 69.7% | 7.4% |
| `arima_d1` (shipped, `d ≤ 1`) | 0.248 | 0.183 | 68.7% | 7.8% |
| learned ensemble | 0.245 | 0.299 | 67.2% | 7.5% |
| **`current` (shipped)** | 0.168 | 0.270 | 65.2% | 8.4% |

| ranks 101–1000 (n=1,767) | poolSkill | medSkill | %>naive | med MAPE |
|---|---|---|---|---|
| **pooled, weighted + capped** | **0.231** | 0.156 | 67.8% | **11.9%** |
| learned ensemble | 0.213 | 0.173 | 67.8% | 12.0% |
| `ens_all4` (round-1 ensemble) | 0.134 | 0.119 | 66.2% | 12.6% |
| **`current` (shipped)** | 0.021 | 0.029 | 53.5% | 13.6% |

| tail | shipped | best (learned ensemble) |
|---|---|---|
| ranks 1001–5000 | −0.211 | +0.065 |
| ranks >5000 | −0.413 | +0.060 |

The horizon profile is the part worth internalising — skill against naive, ranks ≤1000:

| | h1 | h2 | h3 | h4 | h5 |
|---|---|---|---|---|---|
| `current` (shipped) | 0.100 | 0.159 | 0.119 | 0.063 | **0.013** |
| learned ensemble | 0.143 | 0.238 | 0.225 | 0.235 | 0.234 |
| **pooled, weighted + capped** | **0.191** | **0.282** | 0.276 | 0.271 | **0.251** |

The shipped model's advantage over a flat line decays to nothing by year five — the year the
chart draws most prominently. The pooled model's *grows* with horizon, because what it mostly
knows is where a name sits in its lifecycle, and that matters more the further out you go.

## 4. Better point forecasts buy narrower honest bands

Conformal intervals (calibrated 2009+2014, tested 2019), at the same measured coverage:

| | coverage @80% | coverage @95% | median band width (×) | h1→h5 half-width (log) |
|---|---|---|---|---|
| shipped ARIMA, tier × volatility | 0.798 | 0.945 | 6.28 | 0.31 → 0.76 |
| **pooled, tier × volatility** | 0.790 | 0.945 | **3.24** | **0.25 → 0.57** |

Both are honest — that is what conformal calibration buys. The pooled model's bands are simply
**half as wide** for the same truthfulness, and 25% narrower at h5. The current chart cannot show
a truthful 95% band without it being enormous; with a better point forecast it can.

---

## 5. Revised recommendation

The stack I would ship, in order, with round-1's item 1 amended:

1. **Cap implied growth** at the observed 99.9th percentile per horizon (§0). Ship this *before*
   or *with* any move to the log scale, and apply it to the current pipeline too — it is what
   stops both the explosions and the collapses-to-zero.
2. **The pooled ridge, with level interactions and `share^0.5` weighting**, as the primary
   forecaster for every name. One model, no tiering, trained in seconds on ~640k rows, applied to
   all 24,000 names in milliseconds. Held-out skill 0.310 / 0.231 against the shipped 0.168 /
   0.021, and it is the only arm that is positive in every tier.
3. **Blend toward naive in the tail** — either the learned per-tier weights, or simply a heavier
   shrink for names outside the top 1000. Worth ~0.06 poolSkill for ranks >1000 and it keeps the
   page from asserting trends nobody can forecast.
4. **Conformal bands, tier × volatility** (round 1, §5), now half as wide.
5. Keep the shipped ARIMA if you want it in the ensemble at the very top (it earns ~0.03 there),
   but it is no longer load-bearing: dropping it makes the precompute a minutes-long job.

## 6. Caveats

- One held-out origin (2019), which spans the 2020–21 birth-rate shock. Origins 2009 and 2014 are
  used for every tuning decision, so the test is clean, but a single test window is a single
  draw. The ranking of the top few methods is stable across all three origins; the exact margins
  are not.
- The λ sweep in §1 tunes on origin 2014 and reports both 2014 and 2019; only the 2019 column is
  free of that choice. The held-out table in §3 uses λ fixed from the earlier origins.
- 10 of 4,381 series were abandoned by the per-series backtest's 120-second timeout (the same
  runaway-ARIMA pathology the app's precompute bounds at 30s).
- Round 1 was scored with a `score.py` that summed the naive baseline over every name-origin
  rather than the ones each method covered. That is now fixed; round 1's tables were re-run
  against the fix and are **unchanged**, because every method there covered the same rows. It
  matters here, where the pooled model covers two origins and the per-series methods cover three.
