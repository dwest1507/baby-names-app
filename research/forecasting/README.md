# Forecasting research harness

A rolling-origin benchmark for the five-year popularity forecast on `/search`. It exists to
answer one question with numbers instead of intuition: **for the names visitors actually look
at, does a proposed model beat the one we ship, and does it beat assuming nothing changes?**

Nothing here is imported by the app. `methods.py` imports `backend/app/services/forecast.py`
so that the "current" arm of every comparison is the shipped pipeline itself, not a
re-implementation of it.

## Running it

Everything writes to `.work/` (gitignored). Use the backend's virtualenv — these scripts need
numpy/statsmodels:

```bash
cd research/forecasting
PY=../../backend/.venv/bin/python3

$PY extract_series.py                 # data/names.built.db -> .work/series.npz (21,792 series)
$PY backtest.py --top 100000 --mid 1200 --rest 1200 --workers 13   # per-series methods (~20m)
$PY pooled.py   --top 100000 --mid 1200 --rest 1200                # global model (~3m)
$PY pooled.py   --top 100000 --mid 1200 --rest 1200 \
                --train-pool top1000 --name pooled_t1k --out .work/pooled_t1k.jsonl
$PY combine.py .work/main.jsonl .work/pooled.jsonl .work/pooled_t1k.jsonl .work/combo.jsonl
cat .work/main.jsonl .work/pooled*.jsonl .work/combo.jsonl > .work/all.jsonl

$PY score.py      .work/all.jsonl --buckets top100,top1000,top5000,rest --by-horizon
$PY selection.py  .work/all.jsonl          # is per-name model choice worth it? (no)
$PY conformal.py  .work/all.jsonl --method combo_pooled_ens --cal-origins 2014 --test-origin 2019
```

`--top/--mid/--rest` size the evaluation sample per popularity tier; the default run takes
*every* name ranked in 2024's top 1000 and samples the tiers below it.

## What each piece does

| file | role |
|---|---|
| `data.py` | paths, the popularity tiers every table breaks down on, and the stratified sample |
| `cohorts.py` | per-year share held by each ending / initial-letter / sex cohort, leave-one-out |
| `extract_series.py` | pulls each name/sex's observed `(year, popularity_percent)` series out of the DB |
| `methods.py` | the candidate forecasters, including `current` — the shipped ARIMA pipeline |
| `backtest.py` | rolling-origin evaluation; fits each base method once per name-origin and derives the shrunk/ensemble variants from those forecasts |
| `pooled.py` | the global model: one ridge per horizon, learned across all names, in log space |
| `pooled2.py` | the same, with level interactions, cohort features, popularity-weighted fitting and a tuned penalty |
| `pooled3.py` | the same problem fitted with gradient-boosted trees (LightGBM) instead of a ridge, plus the lifecycle feature block |
| `cap.py` | caps a forecast's implied growth at what names actually do, and writes the capped variants |
| `weights.py` | learns ensemble weights per tier and horizon on earlier origins, applies them to the held-out one |
| `combine.py` | log-space combinations of forecasts already produced by the runs above |
| `blend.py` | log-space blend of two named methods, with the weight fitted on earlier origins |
| `score.py` | skill against the naive baseline, by popularity tier and by horizon |
| `selection.py` | picks each name's method on an earlier origin and scores that choice — plus the hindsight oracle |
| `conformal.py` | empirical prediction intervals: calibrate residual quantiles on an earlier origin, measure the coverage they actually achieve |
| `paired.py` | bootstraps the poolSkill difference between two methods, so a gap can be read against its own noise |
| `origins.py` | poolSkill as a method x origin matrix, to check whether a conclusion holds across test windows |
| `quantiles.py` | fits the prediction quantiles themselves with pinball loss, one booster per (alpha, horizon) |
| `intervals.py` | scores band constructions against each other — residual, direct-quantile and conformalised — on interval score, not coverage alone |
| `reconcile.py` | makes the per-name forecasts add up to the share total they have to sum to, and scores what that costs or buys |

## Reading the metrics

* **poolSkill** — `1 - Σ|error| / Σ|naive error|` over a tier. Because it sums *absolute*
  errors, big names dominate it. That is deliberate: it is the popularity-weighted metric.
* **medSkill** — the median of the same ratio computed per name. Every name counts once.
* **%>naive** — share of names where the method beats "assume no change". The honesty check:
  a method can win on poolSkill while losing on most names.
* A method that looks good on one and bad on the other has a fat tail somewhere. Read all three.

The findings are in [FINDINGS.md](FINDINGS.md) (round 1), [FINDINGS-2.md](FINDINGS-2.md)
(round 2 — the untested hypotheses from round 1, plus a correction to one of its
recommendations) [FINDINGS-3.md](FINDINGS-3.md) (round 3 — boosted trees in place of the
pooled ridge), [FINDINGS-4.md](FINDINGS-4.md) (round 4 — the same comparison over 25 origins
instead of one) and [FINDINGS-5.md](FINDINGS-5.md) (round 5 — fitting the prediction quantiles
directly, and making the forecasts add up: both work, neither for the reason expected).
Round 2's pipeline:

```bash
$PY cohorts.py                                       # cohort aggregates for the ablation
$PY backtest.py --top 100000 --mid 1200 --rest 1200 --origins 2009,2014,2019 --timeout 120
$PY pooled2.py --sets inter --weight pop --power 0.5 --lam 100 \
               --top 100000 --mid 1200 --rest 1200 --eval-origins 2009,2014,2019 \
               --name pooled2_pop --out .work/pooled2_pop.jsonl
$PY cap.py .work/main.jsonl .work/pooled2_pop.jsonl --fit-origins 2009,2014
$PY weights.py .work/main.jsonl .work/pooled2_pop.jsonl .work/capped.jsonl \
               --fit-origins 2009,2014 --test-origin 2019
```

Round 3 needs LightGBM, which is not a backend dependency — install it into the same virtualenv
first (`cd backend && uv pip install lightgbm`). Its pipeline:

```bash
# hyperparameters chosen on 2009+2014, scored once on 2019
$PY pooled3.py --model gbt --sets inter --tune --tune-origins 2009,2014 \
               --eval-origins 2019 --name gbt_inter --out .work/gbt_inter.jsonl

# the shipped candidate: base features only, 5 seeds, all three origins
$PY pooled3.py --model gbt --sets "" --leaves 15 --lr 0.03 --trees 300 --min-child 200 \
               --seeds 5 --eval-origins 2009,2014,2019 --name gbt_pop --out .work/gbt_pop.jsonl

# ablations: drop the hand-built interactions, add the lifecycle block, and the same
# feature sets under the ridge, to separate "nonlinearity" from "new features"
$PY pooled3.py --model gbt   --sets ""         ... --name gbt_base
$PY pooled3.py --model gbt   --sets life       ... --name gbt_base_life --importance
$PY pooled3.py --model ridge --sets inter,life --lam 100 ... --name ridge_life

$PY blend.py .work/gbt_pop.jsonl .work/pooled2_pop.jsonl --pair gbt_pop,pooled2_pop \
             --fit-origins 2009,2014 --out .work/blend.jsonl
$PY cap.py .work/gbt_pop.jsonl .work/blend.jsonl --fit-origins 2009,2014 \
           --out .work/capped_gbt.jsonl
$PY paired.py .work/cmp.jsonl --a gbt_pop_cap --b pooled2_pop_cap
$PY conformal.py .work/gbt_pop.jsonl --method gbt_pop --cal-origins 2009,2014 --test-origin 2019
```

Round 4 widens round 3's single held-out origin to 25. `--eval-origins` takes ranges
(`1995:2019`, or `1995:2019:5` with a step), and training rows are evicted between origins, so
peak memory stays at one origin's worth (~0.5 GB) rather than 25.

```bash
# A and B: 25 origins, hyperparameters held fixed and symmetric between the two models
$PY pooled3.py --model gbt   --sets ""      --leaves 15 --lr 0.03 --trees 300 --min-child 200 \
               --eval-origins 1995:2019 --name gbt_pop     --out .work/gbt_many.jsonl
$PY pooled3.py --model ridge --sets inter --lam 100 \
               --eval-origins 1995:2019 --name pooled2_pop --out .work/ridge_many.jsonl

# C: the leakage-free block — tune only on origins whose targets closed before the test starts
$PY pooled3.py --model gbt --sets "" --tune \
               --grid "leaves=7,15,31;lr=0.02,0.03,0.05;trees=150,300,600;min_child=200,1000" \
               --tune-origins 1995,2000,2005,2010 --eval-origins 2015:2019 --name gbt_clean

$PY origins.py .work/gbt_many.jsonl .work/ridge_many.jsonl --common
$PY paired.py  .work/many.jsonl --a gbt_pop --b pooled2_pop   # resamples names, not name-origins
```

`paired.py` and `origins.py` need a `naive` arm only if you ask for one — `origins.py` recomputes
the baseline from each row's `last`, while `paired.py` expects `naive` rows in the file.

Round 5 has two independent halves. The first fits the prediction quantiles instead of pasting
residual quantiles around a point forecast. Seven origins: three to calibrate a band on, four to
score it at, and the calibration block's five-year targets all close before the first test origin
opens, so nothing in the test informs the calibration.

```bash
$PY quantiles.py --alphas 0.025,0.1,0.5,0.9,0.975 \
                 --leaves 15 --lr 0.03 --trees 300 --min-child 200 \
                 --top 100000 --mid 1200 --rest 1200 \
                 --eval-origins 1995,1999,2003,2008,2012,2016,2019 \
                 --name qr --out .work/qr.jsonl

$PY intervals.py .work/qr.jsonl --cal-origins 1995,1999,2003 \
                 --test-origins 2008,2012,2016,2019 --conditional --by-horizon
```

The quantile objective is roughly ten times slower per booster than squared error, because
LightGBM recomputes every leaf value as a weighted percentile — budget hours, not minutes, and
run it alone on the machine.

The second half needs forecasts for the *whole* eligible set, not the stratified sample, because
the constraint being imposed is a sum over every name:

```bash
$PY pooled3.py --model gbt --sets "" --leaves 15 --lr 0.03 --trees 300 --min-child 200 \
               --top 100000 --mid 100000 --rest 100000 --eval-origins 1995:2019 \
               --name gbt_pop --out .work/gbt_full.jsonl

$PY reconcile.py .work/gbt_full.jsonl --out .work/recon.jsonl   # drift table + 3x3 arms
$PY reconcile.py .work/gbt_full.jsonl --set top1000             # constraint inside the top 1000
$PY origins.py   .work/recon.jsonl                              # does it hold by origin?
```
