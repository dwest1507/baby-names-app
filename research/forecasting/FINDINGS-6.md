# Round 6 — the weighting, the past, and the shape of the line

Three of issue #34's open recommendations, taken in the order the issue ranked them:
**1** (refit the prediction quantiles without popularity weights), **2** (recency-weighted
training rows) and **5** (check the five-horizon path is smooth).

Two of the three are negative for the reason the issue gave, and the third — the one the issue
ranked last and framed as a presentation check — is the only change in this round that is worth
shipping. It also turns out not to be a presentation change at all.

**Summary.**

| recommendation | verdict | number |
|---|---|---|
| 1. quantiles without popularity weights | **no** — the tier split is not the weighting | mid-tier miss-high 14.2% -> 13.7%, and −0.039 top-100 poolSkill on the point model |
| 2. recency-weighted training rows | **no** — monotone degradation as the decay sharpens | best half-life beats "off" by **0.0002** |
| 5. smooth the five-horizon path | **yes** — and it is an accuracy change, not a cosmetic one | **+0.0024 / +0.0023 / +0.0012** poolSkill in the top three tiers, all P=100% |

Both negatives are negatives *for the reason the issue gave*, which is the same pattern round 5
reported: the issue has been good at spotting defects and less good at diagnosing them. The
mid-tier band really is too narrow on the upside and the fit really does over-weight popular
names; those two facts are just not connected. Recency really is unmodelled and the 1930s really
are a different era; the pooled model just does not care.

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

Swept on **origins 1995, 2000, 2005 and 2010** — round 4's leakage-free tuning block, whose
five-year targets all close before the recent origins open — with every other hyperparameter
pinned. `half_life=1000000` is the **off** arm: the decay is exactly 1 over the 85 years the pool
spans, so it reproduces the round-4 model inside the same sweep and the same process.

| half-life (years) | 10 | 20 | 40 | 80 | **off** |
|---|---|---|---|---|---|
| poolSkill, top 100 + ranks 101-1000 | 0.3166 | 0.3182 | **0.3200** | 0.3191 | 0.3198 |

**The answer is no, and the shape of the no is informative.** Skill degrades monotonically as the
decay gets sharper — 40 to 20 to 10 loses 0.0018 then 0.0016 — which is what a model losing
training data looks like, not what a model shedding irrelevant training data looks like. The
best arm beats **off** by **0.0002**, an order of magnitude below the smallest effect this
project has been willing to call real (round 5's reconciliation, at 0.001-0.007) and two orders
below round 3's model-class change (0.055). The tuner picks 40 because something has to win a
five-way comparison; nothing here distinguishes it from the flat arm.

The precedent does not transfer, and it is worth saying why. Round 2's trailing window helped
*exponential smoothing*, which fits one series and has a handful of effective parameters, so
every year of stale history is a year of biased slope estimate. The pooled model fits **640,000
name-origins** and learns a mapping from trajectory shape to future growth. A 1935 name-origin
that rose 30% over five years and then gave half of it back is teaching the same lesson as a 2014
one. Recency mattered for the local fit because the *level* was stale; it does not matter for the
global fit because the *relationship* is not.

This is the third independent finding pointing the same way. Round 3 found more model class does
not help, round 4 found more origins do not change the tuning, and this finds more recent data
does not beat more data. The pooled model is not short of anything the name's own history can
supply.


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

## New dead ends for the list

- **`--weight none` on the quantile fit.** Does not close the tier split (mid-tier miss-high
  14.2% -> 13.7%) and costs 0.039 of top-100 poolSkill on the point forecast. The weighting is
  not what makes the mid-tier band wrong.
- **Recency weighting, in both forms.** A geometric decay on training-row age degrades
  monotonically as it sharpens; the best half-life is worth 0.0002 over no decay at all.
- **A straight line in log space** as the forecast path. Smoothest possible, wins ranks
  101-1000 by +0.0019, and is the worst arm in both tail tiers (−0.0104 in `rest`) because it
  cannot express flattening-out. The path's bend carries information; only its corners do not.
- **Correcting horizon 1 alone.** The h1 under-move is real (48% of the actual mean step against
  63-72% later) but fixing only that point gets +0.0008 on the top 100 against the full
  smoother's +0.0024, and loses both tail tiers.

## What this changes about shipping

Round 5's list stands, with one addition and one deletion:

- **Add:** smooth the path with a three-point moving average over the log steps before drawing
  it, and before reconciling. Endpoint-preserving, so the five-year number does not move; worth
  +0.001-0.002 in the three tiers that matter and −0.0009 in the deep tail; costs nothing at
  runtime.
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
