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
| `extract_series.py` | pulls each name/sex's observed `(year, popularity_percent)` series out of the DB |
| `methods.py` | the candidate forecasters, including `current` — the shipped ARIMA pipeline |
| `backtest.py` | rolling-origin evaluation; fits each base method once per name-origin and derives the shrunk/ensemble variants from those forecasts |
| `pooled.py` | the global model: one ridge per horizon, learned across all names, in log space |
| `combine.py` | log-space combinations of forecasts already produced by the runs above |
| `score.py` | skill against the naive baseline, by popularity tier and by horizon |
| `selection.py` | picks each name's method on an earlier origin and scores that choice — plus the hindsight oracle |
| `conformal.py` | empirical prediction intervals: calibrate residual quantiles on an earlier origin, measure the coverage they actually achieve |

## Reading the metrics

* **poolSkill** — `1 - Σ|error| / Σ|naive error|` over a tier. Because it sums *absolute*
  errors, big names dominate it. That is deliberate: it is the popularity-weighted metric.
* **medSkill** — the median of the same ratio computed per name. Every name counts once.
* **%>naive** — share of names where the method beats "assume no change". The honesty check:
  a method can win on poolSkill while losing on most names.
* A method that looks good on one and bad on the other has a fat tail somewhere. Read all three.

The findings from the run of 2026-09-05 are in [FINDINGS.md](FINDINGS.md).
