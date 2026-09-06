# Round 3: the nonlinear pooled model

Follow-up to [FINDINGS.md](FINDINGS.md) and [FINDINGS-2.md](FINDINGS-2.md), run 2026-09-05
against the same database. This round tests recommendation 1 of
[issue #34](https://github.com/dwest1507/baby-names-app/issues/34) — replace the pooled ridge
with gradient-boosted trees — plus recommendation 2 (lifecycle features) and a piece of
recommendation 4 (more origins).

The discipline is unchanged from round 2. Hyperparameters, loss, weighting, blend weights and
growth caps are all chosen on origins **2009 and 2014**; origin **2019** (forecasting 2020–2024)
is the held-out test and is touched exactly once per arm. Training rows for an origin are the
name-origins whose five-year targets had already been observed by then.

**Result in one line: yes, the response surface is nonlinear, and the trees find it.** The
boosted model wins the top 100 by +0.042 poolSkill (99% of bootstrap draws), wins both tail
tiers, has better median skill in every tier — and is 0.011 behind the ridge in ranks 101–1000,
which is the one place the issue's success criterion asked it to win.

---

## 1. Setup

`pooled3.py`, LightGBM 4.7.0. Everything except the model class is held fixed so the comparison
is like-for-like, which is what `--model ridge` in the same script is for:

- the same feature builder (`pooled2.features`) — `pooled2.py` gained an additive `extra=`
  hook for the lifecycle block and is otherwise unchanged; the round-2 arm re-runs
  byte-identical
- the same target: `log(y_{t+h} / y_t)`, one model per horizon, five horizons
- the same `share^0.5` popularity weights, clipped at 50× the mean
- the same evaluation sample, tiers and metrics

## 2. Hyperparameters want *less* capacity, and the answer is flat in them

Coordinate descent, scored as mean poolSkill over ranks 1–100 and 101–1000, on the two tuning
origins only. Two passes: the first from LightGBM-ish defaults, the second restarted from its
winner because it had hit the low edge of the grid on both `leaves` and `trees`.

| pass 1 — start `leaves 63, lr 0.05, 600 trees, min_child 200` → 0.2919 | |
|---|---|
| leaves | **31 → 0.2979**, 63 → 0.2919, 127 → 0.2833, 255 → 0.2708 |
| learning rate | **0.03 → 0.3031**, 0.05 → 0.2979, 0.1 → 0.2854 |
| trees | **300 → 0.3068**, 600 → 0.3031, 1000 → 0.2993 |
| min_child_samples | 50 → 0.3067, **200 → 0.3068** (kept), 1000 → 0.3066 |

| pass 2 — start `leaves 31, lr 0.03, 300 trees, min_child 200` → 0.3068 | |
|---|---|
| leaves | 7 → 0.2994, **15 → 0.3083**, 31 → 0.3068 |
| learning rate | 0.01 → 0.2886, 0.02 → 0.3063, **0.03 → 0.3083** (kept) |
| trees | 150 → 0.3003, **300 → 0.3083** (kept), 600 → 0.3058 |
| min_child_samples | **200 → 0.3083** (kept), 2000 → 0.3082 |

**Chosen: 15 leaves, lr 0.03, 300 trees, `min_child_samples` 200.**

Two things matter more than the winning row. First, every knob wanted a *smaller* model than the
usual defaults — pass 1 hit the low edge on both `leaves` and `trees`, which is why there is a
pass 2. The signal in this problem is smooth, which is also why the ridge was competitive at all.
Second, the whole sweep spans 0.271–0.308 and the top eight configurations sit inside 0.003 of
each other: the held-out result below is **0.354 / 0.220** for the tuned setting and
**0.354 / 0.226** for pass 1's coarser `leaves=31` winner. Nothing here is knife-edge, so none of
it is a tuning artifact.

## 3. Held-out result (origin 2019, everything chosen on 2009/2014)

`gbt_pop_cap` is 5 seeds averaged in log space; `pooled2_pop_cap` is round 2's winner.

| ranks 1–100 (n=198) | poolSkill | medSkill | %>naive | medMAPE |
|---|---|---|---|---|
| **gbt (this round)** | **0.352** | **0.377** | 70.7% | **7.1%** |
| pooled ridge (round 2) | 0.310 | 0.310 | 68.7% | 7.3% |
| `current_s70_cap` | 0.255 | 0.285 | 69.7% | 7.4% |
| **`current` (shipped)** | 0.168 | 0.270 | 65.2% | 8.4% |

| ranks 101–1000 (n=1,767) | poolSkill | medSkill | %>naive | medMAPE |
|---|---|---|---|---|
| pooled ridge (round 2) | **0.231** | 0.156 | 67.8% | 11.9% |
| **gbt (this round)** | 0.220 | **0.196** | 63.8% | **11.7%** |
| learned ensemble | 0.213 | 0.173 | 67.8% | 12.0% |
| **`current` (shipped)** | 0.021 | 0.029 | 53.5% | 13.6% |

| tail | gbt | pooled ridge | shipped |
|---|---|---|---|
| ranks 1001–5000 | **0.079** | 0.064 | −0.211 |
| ranks >5000 | **0.079** | 0.032 | −0.413 |

Bootstrapping the name-origins (`paired.py`, 5,000 draws, both methods on the same resample):

| tier | gbt | ridge | difference | P(gbt better) |
|---|---|---|---|---|
| ranks 1–100 | +0.352 | +0.310 | **+0.042 [+0.007, +0.077]** | 99% |
| ranks 101–1000 | +0.220 | +0.231 | −0.011 [−0.030, +0.007] | 12% |
| 1001–5000 | +0.079 | +0.064 | +0.016 [+0.000, +0.032] | 98% |
| >5000 | +0.079 | +0.032 | **+0.047 [+0.033, +0.062]** | 100% |

Against the shipped model the gap is not close: **+0.184 [+0.117, +0.258]** on the top 100 and
**+0.199 [+0.157, +0.246]** on ranks 101–1000, both at P=100%.

**The issue's success criterion is met on ranks 1–100 and on both tail tiers, and missed on ranks
101–1000.** The miss is 0.011 of poolSkill with a confidence interval straddling zero, in the one
metric where a handful of large names can move the total — and in that same tier the boosted
model has the better *median* skill (0.196 vs 0.156), the better median MAPE and the better
median log error. It wins the typical name there and loses a few big ones.

### It is not a one-origin result

Per-origin poolSkill, gbt vs ridge (2009 and 2014 are the tuning origins for both, so only 2019
is clean for either):

| origin | 1–100 | 101–1000 | 1001–5000 | >5000 |
|---|---|---|---|---|
| 2009 | **0.272** / 0.185 | **0.273** / 0.242 | **0.033** / 0.000 | **−0.036** / −0.098 |
| 2014 | **0.448** / 0.387 | **0.238** / 0.237 | **0.116** / 0.085 | **0.063** / −0.023 |
| 2019 | **0.357** / 0.313 | 0.220 / **0.230** | **0.078** / 0.063 | **0.077** / 0.030 |

The boosted model wins **11 of 12** origin × tier cells. The single loss is the 2019
101–1000 cell above.

## 4. The hand-built interactions were a manual approximation of exactly this

Round 2's biggest structural gain came from adding one interaction family by hand
(`level × {g1,g3,g5,g10,accel}`). Give the trees the plain feature block and they reproduce it:

| feature set (origin 2019, poolSkill) | 1–100 | 101–1000 | 1001–5000 | >5000 |
|---|---|---|---|---|
| ridge, base features (round 2) | +0.005 | +0.107 | +0.079 | +0.014 |
| ridge, + level interactions (round 2) | +0.190 | +0.197 | +0.085 | +0.033 |
| ridge, + interactions, weighted (round 2) | +0.313 | **+0.230** | +0.063 | +0.030 |
| **gbt, base features only** | **+0.356** | +0.219 | +0.078 | +0.077 |
| gbt, + level interactions | +0.354 | +0.220 | +0.079 | +0.077 |

Hand-built interactions are worth **+0.185** to the ridge (+0.005 → +0.190, before the weighting
that adds another +0.123) and **nothing** to the trees (+0.356 → +0.354, inside seed noise).
That is the cleanest confirmation available that what the ridge was missing was interaction
structure, not information: the same 13 columns carry it once the model can bend them.

(Ablation tables in §4–§5 are each arm's own score over the 200 top-100 name-origins it covers,
single seed, uncapped. §3's table is `score.py` over the 198 that *every* method in the
comparison covers, which is why the same model reads 0.356 here and 0.352 there.)

The gain profile agrees. `lvl_g5` takes 28% of h5 gain when the interaction columns are present;
remove them and `g5` and `level` simply absorb it.

## 5. Lifecycle features: measured, and they do not help the names that matter

Recommendation 2's features, built in `pooled3.Lifecycle` — rise slope and rise duration, fall
slope, current slope as a fraction of the rise slope, years since the name last held half its
peak share, recent-vs-historic volatility ratio, trailing flat years, and the cross-sex block
(the same name in the other sex: presence, level, 3- and 5-year growth, and this sex's share of
the pair).

| origin 2019, poolSkill | 1–100 | 101–1000 | 1001–5000 | >5000 |
|---|---|---|---|---|
| gbt, base | **+0.356** | **+0.219** | +0.078 | +0.077 |
| gbt, base + lifecycle | +0.333 | +0.209 | **+0.080** | **+0.083** |
| gbt, + interactions + lifecycle | +0.325 | +0.210 | +0.080 | +0.083 |
| ridge, + interactions (round 2) | +0.313 | **+0.230** | +0.063 | +0.030 |
| ridge, + interactions + lifecycle | +0.189 | +0.211 | +0.060 | +0.036 |

They **cost** 0.023 on the top 100 and 0.010 on ranks 101–1000, and buy 0.002/0.006 in the tail.
They are not inert — `slope_ratio` takes 11.8% of h5 gain, second only to `g5`, and `fall_slope`
another 2.9% — they are simply *re-describing* what `g5`, `level`, `below_peak` and
`yrs_since_peak` already say, at the cost of 12 more columns to overfit through. For the ridge
they are actively destructive (0.313 → 0.189).

This is the same result letter-level cohorts got in round 2, and for the same reason: the name's
own recent trajectory already encodes it. **Recommendation 2 is a dead end as specified.** The
cross-sex block is the part with an argument for a second look — it is genuinely new information
rather than a re-parameterisation — but it is buried inside a block that loses on net, and
isolating it was not worth another origin's compute given the size of the effect.

## 6. Three things that did not need doing after all

**Blending the ridge back in.** `blend.py` picks a log-space weight on 2009+2014. The sweep is
monotone all the way to the boundary — 0.2557 at w=0 (pure ridge), 0.2872 at w=0.5, **0.2941 at
w=1.0** (pure gbt) — so the fitted blend *is* the boosted model. Whatever the ridge knows, the
trees already know. This also means the round-2 ensemble machinery (`weights.py`) has nothing
left to combine at the top of the site.

**The growth cap.** Round 2 made the cap mandatory: the log-scale ARIMA arm produced five-year
ratios up to 2.5e44. Applying the same cap to the boosted forecasts clips **0 of 22,534**
forecasts. Trees cannot extrapolate — a leaf value is an average of training targets, so the
model is structurally incapable of the divergence the cap exists to catch. Keep the cap as a
cheap guard rail; stop treating it as load-bearing.

**Seed averaging.** Five seeds averaged in log space score 0.357 / 0.220 against a single seed's
0.356 / 0.219. Subsampling noise is not a factor at this data size; one seed is enough, and the
five-seed arm is reported above only because it was already run.

## 7. The loss is mismatched to the metric, and fixing it does not pay

poolSkill sums absolute errors in *share* space, while the model minimises squared error in *log*
space. Since `|Δshare| ≈ share · |Δlog|`, the loss that matches the metric is L1 in log space
weighted by `share^1`, not L2 weighted by `share^0.5`. Tested on the tuning origins:

| loss × weight | 2009 t100 | 2009 t1k | 2014 t100 | 2014 t1k | mean |
|---|---|---|---|---|---|
| **L2 × share^0.5** (shipped setting) | 0.271 | 0.273 | 0.446 | 0.237 | 0.3067 |
| L2 × share^1 | 0.264 | 0.278 | 0.444 | 0.242 | **0.3070** |
| L1 × share^0.5 | 0.246 | 0.278 | 0.426 | 0.243 | 0.2983 |
| L1 × share^1 | 0.228 | 0.276 | 0.433 | 0.245 | 0.2955 |

The mechanism is real and points the predicted way — **every** move toward L1 or toward heavier
weighting improves ranks 101–1000 (0.273 → 0.276, 0.237 → 0.245) — and every one of them costs
more than that at the top (0.271 → 0.228, 0.446 → 0.426). So the 101–1000 gap in §3 is
*purchasable*, at a worse price than it is worth. `L2 × share^0.5` stays, which also keeps the
setting identical to the ridge's and the comparison like-for-like.

## 8. Bands are slightly narrower, at the same coverage

Conformal intervals (`conformal.py`, calibrated 2009+2014, tested 2019, tier × volatility):

| | coverage @80% | coverage @95% | h1→h5 half-width (log) |
|---|---|---|---|
| pooled ridge | 0.790 | 0.945 | 0.25 → 0.57 |
| **gbt** | 0.795 | 0.947 | **0.25 → 0.54** |

A 5% narrower band at h5 for the same truthfulness. Real, but small — the large win here was
round 2's, halving the shipped model's width. Recommendation 3 (fitting the quantiles directly
with pinball loss) is untested and remains open; LightGBM makes it a one-line objective change,
and `pooled3.py --objective quantile` is wired for it.

## 9. What it costs

- **Fit**: one origin end to end — building 636,420 training rows, fitting five boosters,
  forecasting every evaluated name — is **61 s** wall-clock on 16 cores. Four extra seeds add
  45 s, so the boosting itself is roughly 15 s of that and the rest is feature construction.
  Prediction for the whole name set is milliseconds.
- **Dependency**: LightGBM is **9.9 MB** installed (vs. scikit-learn's 35 MB for
  `HistGradientBoostingRegressor`, which needs `joblib`, `threadpoolctl` and `scipy` besides).
- **Deploy size: zero.** Forecasts are a build artifact (ADR 0004) and `backend/Dockerfile`'s
  runtime stage runs `uv sync --frozen --no-dev` and copies only `app/`, not `scripts/`. A
  `[dependency-groups] precompute` entry is installed by neither. Even in the worst case — a
  plain runtime dependency — 9.9 MB against a measured 907 MB image is +1.1%.
- **Precompute**: this *replaces* a 48-order AICc grid search per name, run twice per name, with
  one ~15 s fit for the whole corpus. It makes the batch dramatically cheaper, not more
  expensive.

## 10. Recommendation

**Ship the boosted pooled model as the primary forecaster**, on the round-2 stack, replacing the
ridge at step 2:

1. Growth cap (round 2 §0) — keep, now as a guard rail rather than a fix. It never fires on this
   model and it still protects the tail cases of anything else in the pipeline.
2. **LightGBM, 15 leaves, lr 0.03, 300 trees, `min_child_samples` 200, one booster per horizon,
   `share^0.5` popularity weights, base features only** — no hand-built interactions, no
   lifecycle features, no cohort features, one seed. It is the best arm in three of four tiers,
   the best on median skill in all four, and the only one with better than −0.05 anywhere in the
   tail.
3. Tail blending toward naive (round 2 §5.3) — still worth having, but less urgently: the tail
   tiers are now +0.079 rather than +0.03.
4. Conformal bands, tier × volatility, 5% narrower.

The 101–1000 poolSkill gap is the honest caveat and it is 0.011 wide. If it has to be closed, §7
says how — L2 × `share^1` buys 0.005 of it for 0.005 off the top 100 — but a like-for-like arm
that wins 11 of 12 origin × tier cells and every median is the one to ship.

## 11. Caveats

- One held-out origin, as in round 2, and it still spans the 2020–21 birth-rate shock.
  Recommendation 4 (eight to ten origins) is only partly done here: §3 reports three origins, but
  two of them are the tuning origins, so they are a consistency check rather than three
  independent tests. The full sweep remains the cheapest way to tighten every number above.
- The hyperparameter sweep is coordinate descent from one starting point, not a grid. §2 argues
  from the flatness of the response that this does not matter; it is not a proof.
- The lifecycle ablation holds the hyperparameters at the values tuned for the base feature set.
  A feature block with 12 more columns might prefer a different capacity, though the direction of
  the miss (worse at the top, better in the tail — the classic overfit signature) suggests
  retuning would shrink the gap rather than reverse it.
- `x_frac` and the rest of the cross-sex block were never ablated on their own. See §5.
