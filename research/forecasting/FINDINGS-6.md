# Round 6 — the weighting, the past, and the shape of the line

Three of issue #34's open recommendations, taken in the order the issue ranked them:
**1** (refit the prediction quantiles without popularity weights), **2** (recency-weighted
training rows) and **5** (check the five-horizon path is smooth).

One is negative, two are positive, and none of the three works the way the issue said it would.
The one the issue ranked first fails outright. The one it ranked second works — but only at
recent origins, which is why the tuning discipline this project has been using could not see it.
The one it ranked last and framed as a presentation check is the largest of the three and is not
a presentation change at all.

**Summary.**

| recommendation | verdict | number |
|---|---|---|
| 1. quantiles without popularity weights | **no** — the tier split is not the weighting | mid-tier miss-high 14.2% -> 13.7%, and −0.039 top-100 poolSkill on the point model |
| 2. recency-weighted training rows | **yes, small** — but invisible to the tuning block that chose it | **+0.003 / +0.008 / +0.006** in the lower three tiers, P=100%, 21 / 25 origins |
| 5. smooth the five-horizon path | **yes** — and it is an accuracy change, not a cosmetic one | **+0.0024 / +0.0023 / +0.0012** poolSkill in the top three tiers, all P=100% |

The methodological finding matters more than any of the three. **A four-origin tuning block put
recency's measured value at +0.0007 when its true value over 25 origins is +0.0025**, because
the effect grows with the origin and the block sits in the era where it does not exist. Round 4
established tuning on early origins as the leakage-free discipline; it is the wrong place to
measure any parameter whose job depends on how much history exists.

All three repeat round 5's pattern: the issue has been good at spotting defects and less good at
diagnosing them. The mid-tier band really is too narrow on the upside, and the fit really does
over-weight popular names — those two facts are simply not connected, and removing the weights
leaves the defect where it was. Recency really does help, but not because the 1930s are a
different process: sharper decay makes things worse, and the winning half-life leaves the 1930s
at roughly a quarter weight rather than dropping them. The model wants the old data, it just
wants it to count for less.

Recommendation 5 was ranked last and framed as a presentation check — "worth a histogram of
second differences before anything ships". It is the only change in this round worth shipping,
it is free at runtime, and it does not trade one tier against another except in the deep tail.


## 1. Refitting the quantiles without popularity weights (recommendation 1)

Round 5 fitted the prediction quantiles directly with pinball loss and found a **tier split**:
`direct` beats the incumbent residual band on the top 100 and loses ranks 101-1000. The issue's
diagnosis was that the loss is weighted by `share^0.5`, so mid-tier names get too little say in
the fit and their upper quantile comes out too low — `direct` misses **high** 14.2% of the time
against a 10% nominal. One run of `quantiles.py --weight none` was supposed to close it.

It does not. Same seven origins, same three calibration origins, same four test origins, same
rows — the only thing that changes between the two fits is the weight vector:

**Interval score at nominal 80% (lower is better), `direct` minus the residual band, cluster
bootstrap over names:**

| tier | weighted `share^0.5` | **unweighted** |
|---|---|---|
| ranks 1-100 | +0.039 [+0.011, +0.067] P=100% | **+0.068 [+0.040, +0.096] P=100%** |
| ranks 101-1000 | −0.042 [−0.054, −0.028] P=0% | **−0.032 [−0.044, −0.019] P=0%** |
| 1001-5000 | −0.001 [−0.011, +0.009] P=42% | +0.003 [−0.006, +0.012] P=73% |
| >5000 | −0.056 [−0.066, −0.045] P=0% | −0.056 [−0.067, −0.045] P=0% |

At nominal 95% the picture is the same: the top-100 win grows (+0.139 → +0.161) and ranks
101-1000 still lose at P=1%.

**The mid-tier defect barely moves.** Its miss-high rate goes from **14.2% to 13.7%** against a
10% nominal — half a percentage point, on a defect of four. The upper quantile for mid-tier names
is not too low *because those names were down-weighted*; dropping the weights entirely leaves it
almost exactly where it was. Whatever is wrong there is a property of the fit, not of the loss's
weighting.

**And it costs more than it buys.** The unweighted fit is not free — it is the same change applied
to the point model, and there the damage is plain:

| point forecast, poolSkill vs naive | top 100 | 101-1000 | 1001-5000 | >5000 |
|---|---|---|---|---|
| L2 mean, `share^0.5` weights | **0.317** | 0.265 | 0.073 | 0.028 |
| L2 mean, unweighted | 0.278 | 0.247 | 0.066 | 0.022 |
| q50 median, unweighted | 0.297 | 0.263 | 0.091 | 0.034 |

Dropping the weights costs **0.039 of top-100 poolSkill** — two thirds of the entire gain round 3
bought by replacing the ridge with boosted trees. The unweighted median recovers about half of
that, which is a mildly interesting aside (the L1-optimal forecast for an absolute-error metric
is the conditional median, and unweighted it does beat the unweighted mean, reversing round 5's
result) but it does not get back to the weighted mean.

**Verdict: no.** The tier split is real, is not caused by the popularity weights, and survives
their removal. Round 5's shipping guidance is unchanged — fitted quantiles for the top 100, the
conformal tier x volatility band elsewhere — and it stays what round 5 admitted it was: a
**post-hoc tier split**, now with one candidate explanation eliminated rather than confirmed.

The obvious remaining suspect is that mid-tier names have a genuinely fatter right tail — a name
ranked 400 can quintuple and a name ranked 20 cannot — and a booster fitting the 90th percentile
of log growth pooled across all names understates it wherever there is the most room above. The
residual band gets that right by construction, because its width is read off the empirical
residual spread inside each tier x volatility bin. That is a hypothesis, not a measurement; it
would be tested by fitting the quantiles per tier, not by reweighting them.

## 2. Recency-weighted training rows (recommendation 2)

The training pool reaches back to 1930 and weights a 1935 name-origin exactly like a 2014 one.
The issue's argument is that 1940s naming dynamics may simply be a different process, with a
precedent: round 2 found that trailing calendar-year windows helped exponential smoothing
substantially (0.178 -> 0.249 on the top 100).

`pooled2.recency_weights` multiplies each training row by `0.5 ** (age / half_life)`, where age
is how many years before the forecast origin that row's own origin sits, normalised to mean 1 so
it composes with the `share^0.5` popularity weights without changing the effective sample size.
`--window` is the hard-cutoff form: keep only the most recent W training origins.

### The sweep says nothing. The held-out test says yes.

Swept on **origins 1995, 2000, 2005 and 2010** — round 4's leakage-free tuning block — with every
other hyperparameter pinned. `half_life=1000000` is the **off** arm: the decay is exactly 1 over
the 85 years the pool spans, so it reproduces the round-4 model inside the same sweep.

| half-life (years) | 10 | 20 | 40 | 80 | **off** |
|---|---|---|---|---|---|
| poolSkill, top 100 + ranks 101-1000 | 0.3166 | 0.3182 | **0.3200** | 0.3191 | 0.3198 |

Read on its own this looks like a dead end: the winner beats **off** by 0.0002, and skill degrades
monotonically as the decay sharpens past 40. **It is not a dead end.** Scoring the chosen
`half_life=40` against the round-4 model over all 25 origins — same hyperparameters, same seed,
same evaluation sample, the weight vector the only difference — gives a small, consistent,
significant gain:

| tier | half_life=40 | round-4 model | difference | origins won |
|---|---|---|---|---|
| ranks 1-100 | 0.347 | 0.345 | +0.001 [−0.001, +0.003] P=90% | — |
| ranks 101-1000 | **0.283** | 0.279 | **+0.003 [+0.002, +0.004] P=100%** | — |
| 1001-5000 | **0.109** | 0.101 | **+0.008 [+0.007, +0.010] P=100%** | — |
| >5000 | **0.045** | 0.039 | **+0.006 [+0.005, +0.008] P=100%** | — |
| top 100 + 101-1000 | 0.306 | 0.304 | +0.0025 | **21 / 25** |

Positive in all four tiers, significant in three, and it wins 21 of 25 origins on the combined
top-1000 metric. That is the same magnitude and the same shape as round 5's reconciliation
(+0.001 to +0.007, positive everywhere) — which this project accepted as real.

### Why the sweep missed it, and the methodological lesson

The gain is **not constant across origins. It grows with the origin**, on the combined
top-1000 metric:

| origin block | gain |
|---|---|
| 1995-2002 | +0.0003 |
| 2003-2010 | +0.0031 |
| 2011-2019 | +0.0040 |
| **the four tuning origins (1995, 2000, 2005, 2010)** | **+0.0007** |
| all 25 | +0.0025 |

The tuning block sits almost entirely in the era where the effect does not exist, so the sweep
measured a real effect at roughly a sixth of its average size and could not separate it from
noise. The mechanism is mechanical: at origin 1995 the training pool spans 1930-1990 and a
40-year half-life barely discriminates inside it; at origin 2019 it spans 1930-2014 and the same
half-life is genuinely down-weighting the 1930s and 1940s. **A recency parameter has less and
less to do the further back the origin sits.**

That is a trap in this harness's own conventions. Round 4 established tuning on early origins as
the leakage-free discipline, and it is the right discipline for every knob whose value does not
depend on how much history exists — but it is exactly the wrong place to measure one that does.
Anything else with the same property (a window, a decay, a burn-in) needs the same treatment:
**tune it early for honesty, then confirm it late, and read the by-origin trend, not the mean.**

### What this does not say

The effect is small, and the top-100 tier — the one the product cares about most — is the one
tier where it is not significant (+0.001, P=90%). The larger tail gains (+0.008, +0.006) are the
tiers with the least product value. So this is worth having and is not worth much: it is a free
multiplication on a weight vector that buys about what reconciliation buys, concentrated in the
places reconciliation already helps.

It also does not vindicate the issue's reasoning. The issue's argument was that the 1940s are a
different *process*. If that were the mechanism, sharper decay should help more, and it does the
opposite — 40 to 20 to 10 loses 0.0018 then 0.0016 in the sweep. A half-life of 40 years over a
pool spanning 85 leaves the 1930s at roughly a quarter weight rather than excluding them. The
model still wants the old data; it just wants it to count for less.

## 3. The shape of the five-year line (recommendation 5)

`pooled3.py` fits one booster per horizon and nothing ties them together: h3's model has never
seen h2's output. A visitor reads the *shape* of the dashed line, so a path that rises, dips and
rises again is a claim the model is not making on purpose. Round 5 checked the equivalent
question for the bands — 0.66% of them narrow as the horizon grows — and said nothing about the
centre line. `smooth.py` is the check, over all 25 origins and 83,547 forecasts.

### The line is not noisy, except in the tail

A "reversal" is a change of direction between consecutive years, ignoring steps below 2% in log
share as invisible. Actual paths are the reference: reality is jagged, and a conditional mean
should come out smoother than a draw from it.

| tier | predicted: reversal% | actual: reversal% | predicted roughness | actual roughness |
|---|---|---|---|---|
| ranks 1-100 | **5.2%** | 34.9% | 0.018 | 0.085 |
| ranks 101-1000 | **8.8%** | 65.3% | 0.020 | 0.154 |
| 1001-5000 | 29.2% | 91.0% | 0.034 | 0.308 |
| >5000 | **43.8%** | 96.5% | 0.051 | 0.472 |

In the two tiers visitors actually look at, the answer to the issue's question is **yes, the path
is smooth** — one path in twenty bends, against one in three for the truth it is predicting. In
the tail it is one in two, which is one more argument for the issue's own open question about
whether the tail should have a five-year line drawn through it at all.

### There is one systematic bend, and it is at h1

The mean year-over-year step, unweighted over every forecast:

| | h1 | h2 | h3 | h4 | h5 |
|---|---|---|---|---|---|
| model | 0.0177 | 0.0245 | 0.0213 | 0.0182 | 0.0153 |
| actual | 0.0370 | 0.0342 | 0.0308 | 0.0277 | 0.0243 |
| model / actual | **0.477** | 0.717 | 0.691 | 0.657 | 0.632 |

The actual path decelerates smoothly. The model's *accelerates* from h1 to h2 and then
decelerates — the first step captures 48% of the move where every later one captures 63-72%. The
line starts flatter than it continues, in every tier, and it shows in the accuracy too: h1 is the
model's **weakest** horizon on the top 100 (poolSkill 0.297, against 0.350 at h5) even though it
is the easiest one to forecast.

(The mean *level* bias in the same table is an artefact of the evaluation sample, not the model:
names are tiered by their 2024 rank at every origin, so today's top-100 names grew on the way
here and any model under-predicts them. FINDINGS-4 flags this. It cancels in every comparison,
because every arm sees the identical sample.)

### Smoothing the path makes it more accurate, not less

Three smoothers, scored on the same poolSkill as everything else. `ma` is a three-point moving
average over the steps with edge padding — which, by construction, **preserves each path's
five-year endpoint exactly** (the padding makes the smoothed steps sum to the raw ones), so it
only redistributes the shape and cannot move the five-year forecast at all.

**poolSkill difference against the raw path, cluster bootstrap over names, 25 origins:**

| tier | `ma` (3-point moving average) | `quad` (least-squares quadratic) | `lin` (straight line) |
|---|---|---|---|
| ranks 1-100 | **+0.0024 [+0.0018, +0.0031] P=100%** | +0.0011 P=100% | +0.0007 P=85% |
| ranks 101-1000 | **+0.0023 [+0.0020, +0.0026] P=100%** | +0.0005 P=100% | +0.0019 P=100% |
| 1001-5000 | **+0.0012 [+0.0009, +0.0016] P=100%** | −0.0005 P=0% | −0.0014 P=0% |
| >5000 | −0.0009 [−0.0014, −0.0004] P=0% | −0.0027 P=0% | −0.0104 P=0% |

`ma` is **positive in the three tiers that matter and costs 0.0009 in the deep tail** — the same
shape as round 5's reconciliation result, and the same order of magnitude (0.001-0.007). It is
not a trade the way most of this project's findings are, because it buys the coherence *and* the
accuracy at once:

| tier | reversal% raw | reversal% after `ma` | roughness raw | after `ma` |
|---|---|---|---|---|
| ranks 1-100 | 5.2% | **2.0%** | 0.0178 | 0.0084 |
| ranks 101-1000 | 8.8% | **3.9%** | 0.0197 | 0.0088 |
| 1001-5000 | 29.2% | 15.8% | 0.0336 | 0.0162 |
| >5000 | 43.8% | 26.0% | 0.0506 | 0.0257 |

Reversals roughly halve and roughness halves, at no cost to the five-year number and a small
gain everywhere above the deep tail.

**Only a third of the gain is the h1 correction.** An arm that moves nothing but the first point
(`h1`, to `ma`'s value) gets +0.0008 on the top 100 against `ma`'s +0.0024, and it *loses* both
tail tiers. So the h1 bend is real and worth fixing, but most of what `ma` buys is genuine
redistribution across the whole path, not one bad horizon.

`lin` — forcing a straight line in log space — is the interesting failure. It is the smoothest
possible path and it wins ranks 101-1000 outright (+0.0019), but it is the **worst** arm in both
tail tiers (−0.0104 in `rest`), because a straight line in log space cannot express the
flattening-out that a tail name's five-year path mostly consists of. The bend is carrying real
information; it is the corner that is not.

### Where it goes in the pipeline

`ma` preserves each path's five-year endpoint but moves h1-h4, so it changes the sums round 5's
reconciler targets at those horizons. The order has to be **smooth first, reconcile second** —
reconciling and then smoothing would break the adding-up the reconciliation just imposed.
`smooth.py --write` emits a smoothed copy of a forecast file so the two can be run in that order
and scored together.

Run that way over all 21,792 names x 25 origins, **the two changes compose**. Reconciliation on
top of the smoothed forecasts still pays, in three of four tiers:

| tier | reconciliation on raw (round 5) | reconciliation on smoothed |
|---|---|---|
| ranks 1-100 | +0.0056 [+0.0029, +0.0085] P=100% | +0.0052 [+0.0024, +0.0080] P=100% |
| ranks 101-1000 | +0.0012 [−0.0002, +0.0027] P=95% | +0.0001 [−0.0014, +0.0015] P=56% |
| 1001-5000 | +0.0037 [+0.0030, +0.0044] P=100% | +0.0024 [+0.0017, +0.0031] P=100% |
| >5000 | +0.0072 [+0.0067, +0.0077] P=100% | +0.0055 [+0.0050, +0.0059] P=100% |

The gains shrink a little, which is what two changes correcting partly-overlapping error should
do, and the stack ends ahead of either alone on the top 100 (poolSkill 0.353 against 0.351 for
reconciliation alone and 0.348 for smoothing alone, from a free forecast at 0.345). Ranks
101-1000 are the exception: smoothing takes that tier most of the way on its own (0.279 -> 0.282)
and reconciliation then has nothing left to remove there.

## New dead ends for the list

- **`--weight none` on the quantile fit.** Does not close the tier split (mid-tier miss-high
  14.2% -> 13.7%) and costs 0.039 of top-100 poolSkill on the point forecast. The weighting is
  not what makes the mid-tier band wrong.
- **Sharp recency decay.** Half-lives below 40 years degrade monotonically (40 -> 20 -> 10 loses
  0.0018 then 0.0016). The 1930s are worth down-weighting, not discarding.
- **A straight line in log space** as the forecast path. Smoothest possible, wins ranks
  101-1000 by +0.0019, and is the worst arm in both tail tiers (−0.0104 in `rest`) because it
  cannot express flattening-out. The path's bend carries information; only its corners do not.
- **Correcting horizon 1 alone.** The h1 under-move is real (48% of the actual mean step against
  63-72% later) but fixing only that point gets +0.0008 on the top 100 against the full
  smoother's +0.0024, and loses both tail tiers.

## What this changes about shipping

Round 5's list stands, with two additions and one deletion:

- **Add:** smooth the path with a three-point moving average over the log steps before drawing
  it, and before reconciling. Endpoint-preserving, so the five-year number does not move; worth
  +0.001-0.002 in the three tiers that matter and −0.0009 in the deep tail; costs nothing at
  runtime.
- **Add, with a caveat:** `--half-life 40` on the training weights. Positive in all four tiers,
  significant in three, 21 of 25 origins, free at runtime — but not significant on the top 100
  (+0.001, P=90%), which is the tier the product cares about most. Worth taking because it costs
  nothing; not worth defending if a future change trades against it.
- **Delete:** "refit the quantiles unweighted" as an open question. It is answered.

## What is left in issue #34

Of the five ranked recommendations, 1, 2 and 5 are now closed. Still open:

- **3. Forecast the corpus total properly** — round 5 showed the `oracle` target roughly doubles
  reconciliation's gain in the top two tiers, so the residual error in the total is a real
  ceiling. Untouched here.
- **4. Isolate the cross-sex block** — the only genuinely new information in the losing lifecycle
  set, never ablated alone. Untouched here.
- Both of the issue's **"two decisions to make"** — whether `poolSkill` is the right target, and
  whether the tail should have a five-year line drawn through it at all. Section 3 adds one small
  piece of evidence to the second: in the tail, the predicted path's skill decays to
  approximately zero by h5 (`rest`: 0.0825 at h1, 0.0057 at h5) while 44% of its paths change
  direction. Whatever is drawn there, the last two years of it are not a forecast.

And one new question, from section 1: **fit the quantiles per tier.** The mid-tier upper quantile
is too low and the weighting is not why. The residual band gets that tail right because its width
is read off the empirical residual spread inside each tier x volatility bin — which is the one
thing the pooled quantile fit does not condition on directly.
