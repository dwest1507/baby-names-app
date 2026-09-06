# Round 4: more origins

Follow-up to [FINDINGS-3.md](FINDINGS-3.md), run 2026-09-06 against the same database. This is
recommendation 1 of the re-ranked list in
[issue #34](https://github.com/dwest1507/baby-names-app/issues/34): every conclusion in rounds
1–3 rested on hyperparameters chosen at two origins and a single held-out test window. This round
runs 25.

**Result in one line: the one place round 3 lost is a single-window artifact.** Over origins
1995–2019 the boosted model beats the pooled ridge in **all four** popularity tiers at P=100%,
including ranks 101–1000, where the 2019-only test had it 0.011 behind.

---

## 1. Design

Three runs, in `.work/many_origins.sh`:

- **A and B — 25 origins, 1995 through 2019, one forecast set per origin per model.** Both models
  keep the hyperparameters they were given in earlier rounds (gbt: 15 leaves, lr 0.03, 300 trees,
  `min_child` 200; ridge: λ=100), both chosen on origins 2009/2014. Applying those to origins
  before 2009 lets the past be scored with a choice made from the future — but it does so
  *symmetrically*, so the comparison between the two is clean even where an absolute level is
  not. Run C is the version without that compromise.
- **C — a clean block.** Hyperparameters re-swept using only origins 1995, 2000, 2005 and 2010
  (whose five-year targets were all observed by 2015), then tested on 2015–2019. Nothing in the
  test block informs anything in the tuning block.

Two harness changes made this possible: `pooled2.evict_train_rows()`, because caching 25 origins
of ~640k training rows at roughly a gigabyte each is an out-of-memory kill, and `--eval-origins
1995:2019` range syntax. Measured peak RSS with eviction: **0.5 GB**, flat across all 25.

`paired.py` now resamples **names**, not name-origins. With 25 overlapping five-year windows per
name, treating each name-origin as an independent draw understates the uncertainty by roughly
half; the interval widths in §2 show both.

## 2. The headline: every tier, every time

Pooled over all 25 origins, with a cluster bootstrap over names (5,000 draws):

| tier | gbt | pooled ridge | difference | P(gbt better) | origins won |
|---|---|---|---|---|---|
| ranks 1–100 | **0.345** | 0.290 | **+0.055 [+0.042, +0.070]** | 100% | 24 / 25 |
| ranks 101–1000 | **0.279** | 0.268 | **+0.011 [+0.003, +0.019]** | 100% | 16 / 25 |
| 1001–5000 | **0.101** | 0.073 | **+0.028 [+0.014, +0.044]** | 100% | 22 / 25 |
| >5000 | **0.039** | −0.024 | **+0.062 [+0.052, +0.074]** | 100% | 25 / 25 |

Round 3's held-out test put the boosted model 0.011 *behind* in ranks 101–1000 (0.220 vs 0.231).
Over 25 origins it is 0.011 *ahead*, and 2019 turns out to be one of only nine origins where the
ridge wins that tier at all. **The success criterion in issue #34 — beat the ridge in ranks 1–100
and 101–1000 without going negative in the tail — is now met in full.**

The ridge is the arm that goes negative: −0.024 pooled in the tail, worse than assuming no change
at all, on 20 of 25 origins. The boosted model is positive in every tier.

The same ordering holds on the per-name metric, which weights every name once instead of by
births — including ranks 1001–5000, where the 2019-only test had the ridge marginally ahead
(0.064 vs 0.062):

| tier | gbt medSkill | ridge medSkill | difference | gbt %>naive | ridge %>naive |
|---|---|---|---|---|---|
| ranks 1–100 | **0.351** | 0.246 | +0.104 | 75.5% | 75.8% |
| ranks 101–1000 | **0.187** | 0.177 | +0.010 | 67.5% | 69.9% |
| 1001–5000 | **0.089** | 0.070 | +0.018 | 60.9% | 58.3% |
| >5000 | **0.045** | 0.008 | +0.036 | 55.9% | 50.9% |

The `%>naive` column is the one place the ridge still holds an edge, in the two top tiers: it
beats a flat line on marginally more names, while losing by more when it loses. That is the same
shape round 3 saw at a single origin, and it is the honest counterweight to the tables above.

Because 25 origins with five-year horizons overlap heavily, here is the same comparison on the
five origins whose test windows do not overlap at all (1995, 2000, 2005, 2010, 2015):

| tier | gbt | ridge | difference |
|---|---|---|---|
| ranks 1–100 | 0.340 | 0.289 | +0.052 |
| ranks 101–1000 | 0.291 | 0.272 | **+0.020** |
| 1001–5000 | 0.116 | 0.083 | +0.033 |
| >5000 | 0.045 | −0.027 | +0.072 |

Same conclusion, and the 101–1000 margin is twice as large.

## 3. Skill over time, and a shock you can see in the metric

Boosted model, poolSkill by origin:

| tier | 1995 | 2000 | 2005 | 2010 | 2015 | 2019 | min | max | mean |
|---|---|---|---|---|---|---|---|---|---|
| ranks 1–100 | 0.399 | 0.365 | 0.340 | 0.260 | 0.356 | 0.356 | 0.256 | 0.446 | 0.347 |
| ranks 101–1000 | 0.278 | 0.330 | 0.300 | 0.288 | 0.260 | 0.219 | 0.206 | 0.353 | 0.278 |

Nothing here is fragile: 25 windows spanning three decades, and the top-100 skill never drops
below 0.256. Round 3's test origin was close to typical rather than a lucky draw — 2019 sits
just above the mean on the top 100 and near the bottom of the range in ranks 101–1000, which is
the honest reading of why that tier looked bad in round 3.

That 101–1000 decline is not noise and not drift. The four worst origins in the whole run are
2016, 2017, 2018 and 2019 (0.265, 0.211, 0.206, 0.219) — precisely the four whose five-year test
windows contain 2020 and 2021. **The birth-rate shock is visible in the metric**, and it hurts
the tier where names are numerous enough for the aggregate to feel it. Any future comparison that
tests only on recent origins is testing partly on how a method handled a pandemic.

A second regime is visible in the tail: at origins 2008–2012, ranks 1001–5000 go slightly
negative for the boosted model (−0.037 to −0.022) and hard negative for the ridge (−0.111 to
−0.080). Those windows cover the post-2008 birth decline. Rare names are where an aggregate shock
shows up first.

## 4. Two origins were enough to choose the hyperparameters

Re-sweeping on four origins (1995, 2000, 2005, 2010) instead of two:

| knob | candidates | outcome |
|---|---|---|
| leaves | 7 → 0.3136, **15 → 0.3200**, 31 → 0.3175 | unchanged |
| learning rate | 0.02 → 0.3183, **0.03 → 0.3200**, 0.05 → 0.3172 | unchanged |
| trees | 150 → 0.3168, **300 → 0.3200**, 600 → 0.3150 | unchanged |
| min_child_samples | 200 → 0.3200, **1000 → 0.3204** | 200 → 1000, worth +0.0004 |

Round 3's choice, made on 2009/2014, survives three of four knobs outright, and the fourth moves
for four ten-thousandths — noise on a surface that round 3 already showed is flat. This is worth
recording as a *negative* result about the original concern: doubling the tuning evidence changed
nothing, so the two-origin tuning in rounds 1–3 was not the weak point it looked like.

## 5. The clean block confirms round 3 directly

Hyperparameters from §4 (chosen on origins ≤2010 only), tested on 2015–2019:

| tier | 2015 | 2016 | 2017 | 2018 | 2019 | pooled |
|---|---|---|---|---|---|---|
| ranks 1–100 | 0.354 | 0.332 | 0.330 | 0.354 | **0.355** | 0.345 |
| ranks 101–1000 | 0.260 | 0.264 | 0.210 | 0.206 | **0.220** | 0.232 |
| 1001–5000 | 0.130 | 0.099 | 0.072 | 0.094 | 0.078 | 0.094 |
| >5000 | 0.092 | 0.101 | 0.090 | 0.080 | 0.077 | 0.088 |

At origin 2019 this scores **0.355 / 0.220** against round 3's 0.356 / 0.219 — with a tuning set
that shares no information with the test at all. The mild leakage in round 3's setup was worth
about one thousandth.

## 6. The growth cap barely moves, and still never fires

Refitting the 99.9th-percentile cap on 20 origins rather than 2:

| | h1 | h2 | h3 | h4 | h5 |
|---|---|---|---|---|---|
| fitted on 2009, 2014 | 1.48 | 2.02 | 2.27 | 2.45 | 2.75 |
| fitted on 1995–2014 | 1.36 | 2.01 | 2.36 | 2.62 | 2.85 |

Either way it clips **0 of 83,547** boosted forecasts. Round 3's finding holds at 25× the sample:
trees cannot extrapolate, so the cap is a guard rail for other model classes, not for this one.

## 7. What this changes

- **The 101–1000 caveat is retired.** The boosted model wins every tier. FINDINGS-3 §3 and §7
  should be read with this round's numbers substituted; the recommendation in FINDINGS-3 §10 is
  otherwise unchanged and now carries no exception.
- **`min_child_samples` 1000** is marginally preferred to 200 on four origins. The difference is
  0.0004; either is defensible.
- **Recommendation 1 in issue #34 is done.** The remaining items — quantile fitting,
  cross-sectional reconciliation, recency-weighted training rows, isolating the cross-sex block,
  the path-smoothness check — are untouched by this round.
- **Test on more than recent origins.** Any future comparison scored only at 2016–2019 is
  measuring pandemic handling as much as forecasting skill.

## 8. Caveats

- **The evaluation sample is tiered by 2024 rank at every origin.** "Ranks 1–100" at origin 1995
  means the names that are top-100 *today*, forecast from 1995. That is the right framing for the
  product — the question is how well the site forecasts the names visitors actually search — but
  it conditions on the future, so absolute skill levels across early origins are flattered. Both
  models see the identical sample, so every comparison in §2 is unaffected. This applies equally
  to rounds 1–3.
- **Overlapping windows.** 25 origins at a five-year horizon are not 25 independent tests. The
  cluster bootstrap handles the correlation between a name's own windows; it does not handle the
  correlation between two names in the same calendar window. §2's non-overlapping subset is the
  conservative view and agrees.
- The shipped ARIMA arm was not re-run over 25 origins — a per-series backtest is ~40 minutes for
  three origins, so this would be hours for no decision-relevant information: it trails by 0.18
  and 0.20 at the one origin where all three were measured.
- Runs A and B apply hyperparameters chosen at 2009/2014 to origins as early as 1995. Symmetric
  between the two models, and §5 gives the leakage-free version.
