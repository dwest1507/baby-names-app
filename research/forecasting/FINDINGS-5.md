# Round 5: fitting the quantiles, and making the forecasts add up

Follow-up to [FINDINGS-4.md](FINDINGS-4.md), run 2026-09-06 against the same database. This
round takes the two remaining structural recommendations from
[issue #34](https://github.com/dwest1507/baby-names-app/issues/34) — fit the prediction
quantiles directly (recommendation 1) and reconcile the forecasts cross-sectionally
(recommendation 2). They are independent, so they are reported as two halves.

**Both recommendations turn out to be right, and both are right for a different reason than the
issue gave.** The directly fitted quantiles win where it matters, but not through the asymmetry
the issue predicted — they are almost perfectly symmetric, and the gain comes from
per-name width instead. Reconciliation is reported in part B.

---

# Part A — fitting the prediction quantiles

## 1. Design

`quantiles.py` fits one LightGBM booster per (alpha, horizon) with the pinball loss, on the
*same* feature block, target, hyperparameters and `share^0.5` popularity weights as the shipped
candidate from round 3. Only the objective changes, so the comparison isolates the loss. Five
alphas — 0.025, 0.1, 0.5, 0.9, 0.975 — give an 80% and a 95% band, plus a median, at a cost of
30 boosters per origin.

Seven origins: **1995, 1999 and 2003 to calibrate any band on, 2008, 2012, 2016 and 2019 to
score it at.** The calibration block's five-year targets all close in 2008, before the first test
window opens, so no band is calibrated on anything its test could see. The test origins were
chosen to span both regime shocks FINDINGS-4 §3 identified: 2008's window covers the post-2008
birth decline and 2019's covers 2020–21.

The L2 arm reproduces FINDINGS-4's run A exactly at every shared origin (0.399 / 0.278 / 0.205 /
0.143 at origin 1995), which is the check that nothing but the loss differs.

`intervals.py` scores six band constructions on identical rows:

| arm | construction |
|---|---|
| `resid_global` | symmetric log-residual quantiles pasted around the point forecast |
| `resid_tiervol` | the same, conditioned on popularity tier x volatility — **round 3's recommendation, the incumbent** |
| `direct` | the quantile boosters' own output, uncalibrated |
| `cqr` | `direct` plus a conformal correction per horizon (Romano et al.) |
| `cqr_tiervol` | the same correction, per tier x volatility |
| `cqr_asym` / `cqr_asym_tier` | the two tails corrected separately |

Coverage alone cannot rank these, because a wider band always covers more. The headline number
is the **Winkler interval score** in log space — width, plus a penalty of `2/(1-level)` times
the distance by which the truth fell outside — which is the proper scoring rule for an interval.
Lower is better.

## 2. The headline: a decisive win on the top 100, a loss in ranks 101–1000

Interval score, 80% and 95% nominal, pooled over the four test origins:

| tier | | `resid_tiervol` (incumbent) | `direct` | difference [95% CI] | P(direct better) |
|---|---|---|---|---|---|
| ranks 1–100 | 80% | 0.735 | **0.696** | **+0.039 [+0.011, +0.067]** | 100% |
| ranks 1–100 | 95% | 1.141 | **1.001** | **+0.139 [+0.066, +0.217]** | 100% |
| ranks 101–1000 | 80% | **0.981** | 1.023 | −0.042 [−0.054, −0.028] | 0% |
| ranks 101–1000 | 95% | **1.493** | 1.543 | −0.050 [−0.085, −0.016] | 0% |
| 1001–5000 | 80% | 1.241 | 1.242 | −0.001 [−0.011, +0.009] | 42% |
| 1001–5000 | 95% | 1.811 | **1.783** | +0.029 [−0.001, +0.058] | 97% |
| >5000 | 80% | **1.431** | 1.487 | −0.056 [−0.066, −0.045] | 0% |
| >5000 | 95% | 2.043 | 2.057 | −0.014 [−0.039, +0.012] | 15% |

Cluster bootstrap over **names**, 5,000 draws, for the reason `paired.py` gives: a name's
overlapping five-year windows are not independent draws.

The top-100 win is large and it is where the product's attention is. At 95% the fitted band is
**17% narrower** (0.734 vs 0.886 in log width) while landing *closer* to nominal coverage —
0.945 against the incumbent's 0.962. At 80% it is 12% narrower at 0.773 against 0.833. The
incumbent buys its coverage by being too wide for popular names, which is exactly what a band
conditioned on a three-way volatility bucket has to do.

Ranks 101–1000 go the other way, and the reason is legible in the miss columns: `direct` misses
*high* 14.2% of the time against a 10% nominal, so its upper quantile is systematically too low
for mid-tier names. It is not a width problem in general — there the fitted band is *wider* on
average than the incumbent (0.912 vs 0.825 at h5) and still scores worse.

Per test origin, top-100 interval score at 80%:

| | 2008 | 2012 | 2016 | 2019 |
|---|---|---|---|---|
| `resid_tiervol` | **0.914** | 0.767 | 0.647 | 0.614 |
| `direct` | 0.975 | **0.753** | **0.548** | **0.508** |

Three of four, and the one loss is 2008 — whose test window is the post-2008 birth decline,
where `direct` misses high 28.1% of the time. The split was already visible in the first half of
the test block (2008, 2012) and holds in the second (2016, 2019), so it is not purely a
post-hoc read.

## 3. Where the win comes from — and it is not asymmetry

Issue #34's argument was that residual quantiles "are symmetric by construction, so they *cannot*
express that a name just past its peak has a fat downside and almost no upside". The premise is
right and **the conclusion is wrong in both directions.**

`asym` — the share of the band's width that sits above the point forecast, where 0.5 is
symmetric:

| arm | ranks 1–100 | 101–1000 | 1001–5000 | >5000 |
|---|---|---|---|---|
| `resid_tiervol` | 0.695 | 0.642 | 0.551 | 0.430 |
| `direct` | 0.488 | 0.498 | 0.494 | 0.492 |

The *residual* band is the strongly asymmetric one — it puts 70% of its width above the point
forecast for top-100 names, because the log-residual distribution is right-skewed. The
**directly fitted quantiles come out almost exactly symmetric.** The mechanism the issue proposed
is not the mechanism that operates.

What the fitted quantiles actually buy is **per-name width**. At horizon 5, within the top 100:

| arm | mean log width | sd of width | distinct widths |
|---|---|---|---|
| `resid_tiervol` | 0.741 | 0.181 | **3** |
| `direct` | 0.673 | 0.352 | **799** |

The incumbent has one width per volatility bucket — three numbers for 800 name-origins. The
fitted quantiles give every name its own, with twice the spread, and on average a *narrower*
one. That is the whole gain: not a band that leans one way, but a band that knows which names are
predictable.

## 4. Conditional coverage is where the incumbent actually fails

A band can have textbook marginal coverage and still be wrong for every name in it. Splitting
the 80% test rows by how far the name sits below its own historic peak at the origin:

| arm | bin | n | coverage | missed low | missed high |
|---|---|---|---|---|---|
| `resid_tiervol` | at peak | 6321 | 0.742 | **15.5%** | 10.3% |
| `resid_tiervol` | off peak | 3581 | 0.769 | 12.5% | 10.6% |
| `resid_tiervol` | long past | 5045 | 0.800 | 9.1% | 10.9% |
| `direct` | at peak | 6321 | 0.798 | 8.8% | 11.4% |
| `direct` | off peak | 3581 | 0.780 | 10.6% | 11.4% |
| `direct` | long past | 5045 | 0.777 | 9.8% | 12.5% |

Nominal is 80% coverage and 10% missed on each side, in *every* bin.

The incumbent's coverage swings 5.8 points across the bins and its low-side miss rate swings 6.4
— it under-covers names at their peak, and it under-covers them by falling short on the
downside, which is precisely the failure issue #34 predicted. `direct` cuts both spreads to
about a third (2.1 and 1.8 points). So the issue identified the right defect and the wrong
remedy: the fix is not asymmetry, it is conditioning.

The same ordering holds at 95%, more mildly (spread 2.5 points against 1.1).

## 5. Conformalising the fitted quantiles adds nothing

| arm | 1–100 | 101–1000 | 1001–5000 | >5000 |
|---|---|---|---|---|
| `direct` | **0.696** | **1.023** | 1.242 | **1.487** |
| `cqr` | 0.699 | 1.024 | 1.242 | 1.490 |
| `cqr_tiervol` | 0.702 | 1.018 | 1.251 | 1.490 |
| `cqr_asym` | 0.691 | 1.000 | 1.248 | 1.587 |
| `cqr_asym_tier` | 0.723 | 1.003 | 1.250 | **1.417** |

At 80%. The conformal correction computed on 1995–2003 is close to zero — the fitted quantiles
are already well calibrated at origins twenty years later — so `cqr` reproduces `direct` to
three decimals. **This is a useful negative result: the pipeline does not need a calibration
step bolted onto the quantile model.**

The asymmetric variant was written specifically to repair the ranks 101–1000 upper tail, since a
symmetric correction cannot fix a band that is honest below and short above. It does move that
tier (1.023 → 1.000) but not far enough to overtake the incumbent's 0.981, and it costs the top
100. `cqr_asym_tier` is the only arm that beats the incumbent in the deep tail
(1.417 vs 1.431, P=100%), which is not worth a pipeline stage on its own.

## 6. Quantile crossing is a non-issue

Issue #34 asked to watch for it. Across the 14,947 test name-origins:

- **0.11%** have quantiles that cross in alpha (a higher alpha predicting a lower value)
- **0.66%** have a band that gets *narrower* as the horizon grows

Both are small enough to fix by sorting, and neither needs a monotone-constrained fit. Five
independently fitted boosters agree with each other essentially all the time.

## 7. The median is not a free upgrade to the point forecast

poolSkill sums *absolute* errors, and the L1-optimal point forecast is the conditional median,
which the alpha=0.5 booster estimates directly — while the shipped model minimises squared error
in log space. So the median ought to be the better point forecast for the metric. It is not,
uniformly:

| | ranks 1–100 | 101–1000 | 1001–5000 | >5000 |
|---|---|---|---|---|
| L2 mean | **0.317** | 0.265 | 0.073 | 0.028 |
| q50 median | 0.300 | **0.273** | **0.094** | **0.039** |

It wins three tiers and loses the one that matters most. Per test origin the top-100 loss is
consistent — the L2 mean wins at 2008 (0.256 vs 0.227), 2012 (0.325 vs 0.311) and 2016
(0.336 vs 0.307), and only 2019 goes the other way (0.356 vs 0.363) — while the median's
ranks-101–1000 win holds at three of four. So this is a real trade with the same shape as every
other one in this research, not noise: the tail and the middle can be bought at the top's
expense. **No change recommended**, but it is worth recording that the loss mismatch
FINDINGS-3 §7 identified is not free money even when the loss is matched exactly.

## 8. What part A changes

- **Recommendation 1 in issue #34 is done, and it lands.** Fitting the quantiles directly is
  worth 12–17% band width on the top 100 at equal or better coverage, and roughly a threefold
  improvement in conditional calibration everywhere.
- **The stated rationale for it was wrong.** Directly fitted quantiles are symmetric; the gain is
  per-name width, not skew. Anything that argued from asymmetry — including the "an honest
  asymmetric band is worth more than a better centre line" framing — should be re-derived.
- **It is a tier-split verdict, not a clean sweep.** `direct` wins ranks 1–100 at P=100% and
  loses ranks 101–1000 at P=100%. Tier is known at serving time, so a tier-switched band is
  implementable, but it is a post-hoc split and should be confirmed on a fresh block before it
  ships.
- **Cost.** 30 boosters per origin instead of 5, and the quantile objective is roughly ten times
  slower per booster than squared error, because LightGBM recomputes every leaf value as a
  weighted percentile. On this machine one origin is ~10 minutes against ~35 seconds. Forecasts
  are a build artifact (ADR 0004), so this is build time, not request time — but it is the one
  place in this stack where the cost is not negligible.

---

# Part B — cross-sectional reconciliation

## 9. Design

`popularity_percent` is a share of one year's births *within a sex*, so it sums to exactly 1
across all names of a sex in a year. That is a hard constraint, not a soft one, and nothing in
this stack has ever imposed it: every forecast is made one name at a time and the sum of them is
free to go wherever it likes. Issue #34's point is that a per-name metric cannot see that error
at all.

Two things had to be got right for the measurement to mean anything:

- **The forecasts have to cover every name.** The stratified evaluation sample is useless for a
  sum. `pooled3.py --top 100000 --mid 100000 --rest 100000 --eval-origins 1995:2019` produces
  **335,135 name-origins over all 21,792 names and 25 origins** — the same model, hyperparameters
  and discipline as FINDINGS-4's run A.
- **The constraint set is the names actually forecast at that origin**, not the corpus. Rows
  whose five-year window is incomplete are dropped from the forecast file, and reconciling to a
  total that included them would build a known shortfall into every factor. `reconcile.py` sums
  each origin-and-sex group's own history out of a dense name x year panel.

The eligible set covers 91–97% of each year's births and its five-year total moves by at most
about 1%, so the adding-up target is real *and* easy to forecast. Three targets — `naive` (the
total stays put), `drift` (damped log-linear extrapolation of the total's own history) and
`oracle` (the total actually observed, as a ceiling) — crossed with three ways to spread the
discrepancy: `prop` (one multiplicative factor, a constant shift in log space), `ols` (the
textbook equal-weight reconciliation) and `vol` (a log shift proportional to each name's own
volatility).

## 10. The drift is real, and it is almost entirely female

`sum(forecast) / sum(actual)`, pooled over all 25 origins and both sexes:

| h1 | h2 | h3 | h4 | h5 |
|---|---|---|---|---|
| 1.0028 | 1.0072 | 1.0105 | 1.0124 | 1.0123 |

The forecasts over-predict the total by 1.2% at five years, and the drift grows monotonically
with the horizon — the signature of a bias, not noise. But the pooled number hides the finding:

| origin | F, h5 | M, h5 | naive F, h5 | naive M, h5 |
|---|---|---|---|---|
| 1995 | 1.0269 | 0.9909 | 1.0165 | 1.0157 |
| 2001 | 1.0365 | 0.9975 | 1.0264 | 1.0158 |
| 2007 | 1.0018 | 0.9829 | 1.0060 | 1.0100 |
| 2013 | 1.0386 | 1.0050 | 1.0041 | 1.0061 |
| 2019 | 1.0168 | 0.9992 | 1.0113 | 1.0082 |

**Female forecasts drift up almost every window; male ones are unbiased.** Across all 25
origins the h5 ratio is above 1.0 at **24 of 25** female origins (mean 1.027, range
0.997–1.042) and at only **8 of 25** male ones (mean 0.998, range 0.979–1.021). At origin 2013
the model over-predicts the female total by 3.9% while persistence over-predicts it by 0.4% —
so the drift is not inherited from the naive baseline, the model adds it.

Splitting h5 by where each name sits relative to its own peak explains where it comes from:

| sex | situation | model sum/actual | naive sum/actual |
|---|---|---|---|
| F | at peak | **1.0385** | 0.8674 |
| F | off peak | **1.0385** | 1.1729 |
| F | long past | 0.9973 | 1.1628 |
| M | at peak | 1.0043 | 0.8698 |
| M | off peak | 1.0094 | 1.1628 |
| M | long past | 0.9792 | 1.1752 |

The model does its job: it fixes almost all of the naive baseline's ±16% error in every bin.
What is left is a **residual 3.9% over-prediction on female names at or just past their peak** —
the fast-churning part of a distribution that is much more diffuse than the male one (the female
top 1000 holds 72% of female births, the male top 1000 holds 81%). That is the whole drift.

## 11. Reconciling removes it, and it helps every tier

poolSkill / medSkill over all 25 origins, whole eligible set:

| arm | ranks 1–100 | 101–1000 | 1001–5000 | >5000 |
|---|---|---|---|---|
| free (round 4's model) | 0.345 / 0.351 | 0.279 / 0.187 | 0.134 / 0.086 | 0.049 / 0.046 |
| **`naive` + `prop`** | **0.351 / 0.357** | **0.281** / 0.186 | 0.137 / 0.084 | 0.056 / 0.048 |
| `drift` + `prop` | 0.350 / 0.356 | 0.281 / 0.186 | **0.138** / 0.085 | 0.057 / 0.048 |
| `naive` + `vol` | 0.349 / 0.354 | 0.280 / 0.188 | 0.135 / 0.080 | 0.058 / 0.047 |
| `naive` + `ols` | 0.346 / 0.352 | 0.279 / 0.192 | 0.122 / 0.041 | 0.014 / −0.022 |
| `oracle` + `prop` *(ceiling)* | 0.353 / 0.351 | 0.287 / 0.180 | 0.145 / 0.083 | 0.067 / 0.052 |

Cluster bootstrap over names, `naive` + `prop` minus the free forecast:

| tier | names | difference [95% CI] | P(better) | origins won |
|---|---|---|---|---|
| ranks 1–100 | 200 | **+0.0056 [+0.0029, +0.0085]** | 100% | 20 / 25 |
| ranks 101–1000 | 1781 | +0.0012 [−0.0002, +0.0027] | 95% | 14 / 25 |
| 1001–5000 | 7318 | **+0.0037 [+0.0030, +0.0044]** | 100% | 17 / 25 |
| >5000 | 11012 | **+0.0072 [+0.0067, +0.0077]** | 100% | 19 / 25 |

**Positive in all four tiers, significant in three, and negative in none.** This is the first
change in five rounds that does not buy one tier at another's expense — because it is not a
trade, it is the removal of a bias.

The gains are small in absolute terms (0.001–0.007 of poolSkill against round 3's 0.055) and
the honest framing is that this is a *coherence* fix that happens to pay a little accuracy, not
an accuracy technique. Two things make it worth having anyway: it costs one multiplication per
forecast, and the sum of what the site displays stops being impossible.

Three secondary results:

- **`ols` is actively harmful** — the textbook equal-weight reconciliation gives every name the
  same absolute share of the discrepancy, which is meaningless when shares span six orders of
  magnitude. It takes ranks >5000 from 0.049 to 0.014 and its medSkill negative. For shares, the
  multiplicative form is not a preference, it is a requirement.
- **`vol` is not worth its complexity.** Letting volatile names absorb more of the correction
  is marginally better in the deep tail (0.058 vs 0.056) and worse everywhere else.
- **Knowing the true total is worth about as much again.** The `oracle` row roughly doubles the
  gain in the top two tiers and triples it in the tail, so the residual error in forecasting the
  total is a real ceiling — but `drift` scores the same as `naive`, so the easy version of that
  improvement is already exhausted.

## 12. Reconciling the top 1000 alone is the weaker version

Issue #34 proposed "forecasting the top 1000 jointly and reconciling to the constraint".
Restricting the constraint set to the top 1000 by 2024 rank:

| tier | difference [95% CI] | P(better) | origins won |
|---|---|---|---|
| ranks 1–100 | +0.0066 [+0.0024, +0.0109] | 100% | 19 / 25 |
| ranks 101–1000 | −0.0005 [−0.0028, +0.0017] | 33% | 11 / 25 |

It helps the top 100 slightly more and does nothing for ranks 101–1000, and by construction it
cannot touch the other two tiers. The reason is visible in its own drift table: pooled over 25
origins the top-1000 sum comes out at **0.9883** of actual at h5 — nearly unbiased, because the
top 1000's aggregate share is itself falling and the model tracks that. **The constraint worth
imposing is the global one.** The top 1000's total is a forecastable quantity; the corpus total
is a fact.

## 13. What part B changes

- **Recommendation 2 in issue #34 is done, and it is worth shipping.** One multiplicative factor
  per (year, sex, horizon), chosen so the forecasts sum to the total observed at the origin.
  Positive in every tier, significant in three, free at runtime.
- **The bias it removes is specific and previously unrecorded**: female names at or just past
  their peak, over-predicted by about 3.9% at five years. Anyone doing further feature work
  should know that is where the model's remaining aggregate error lives.
- **Apply it globally, not per tier.** The corpus-wide constraint beats the top-1000 one in
  every tier but the top 100, where they are within noise of each other.
- **Never use additive reconciliation on shares.**

---

# What is left

From issue #34's list, after this round:

1. ~~Fit the prediction quantiles directly~~ — done, part A. Wins ranks 1–100, loses 101–1000.
2. ~~Cross-sectional reconciliation~~ — done, part B. Wins everywhere, small.
3. **Recency-weighted training rows** — still untested, still cheap, still one parameter.
4. **Isolate the cross-sex block** — still untested. Part B's finding that the drift is
   sex-specific is a mild argument in its favour.
5. **Check the five-horizon path is smooth** — still unlooked-at. Part A's crossing numbers
   (0.66% of bands narrow with the horizon) are a partial answer for the *bands* but say nothing
   about the centre line.

Two new items this round raises:

6. **Refit the quantiles without popularity weights.** `direct` loses ranks 101–1000 by missing
   *high* 14.2% of the time against a 10% nominal — its upper quantile is too low for mid-tier
   names, which is what a `share^0.5`-weighted fit would be expected to do. One run answers it,
   and it is the only obvious route to making the quantile win unconditional.
7. **Forecast the corpus total properly.** The `oracle` row in §11 says a better total is worth
   about as much as reconciliation itself, and `drift` shows the trivial version does not get
   there.

## Caveats

- **Part A's test block is four origins, part B's is twenty-five.** The quantile objective is
  ~10x the cost of squared error per booster, so the two halves are not on equal evidence. The
  interval conclusions should be read as four windows, with a cluster bootstrap over names inside
  them, not as FINDINGS-4's twenty-five.
- **The tier split in part A is post-hoc.** It was visible in the first half of the test block
  and held in the second, which is suggestive but is not a pre-registered confirmation.
- The evaluation sample is tiered by 2024 rank at every origin, as in every previous round; see
  FINDINGS-4 §8. Both halves of every comparison see the identical sample.
- Part B's `naive` target is itself slightly biased — persistence over-predicts the female total
  by up to 2.6% at h5 — so §11's gains are achieved *despite* an imperfect target, which is what
  the `oracle` row is there to bound.
