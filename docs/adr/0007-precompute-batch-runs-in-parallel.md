# 7. The precompute batch runs in parallel, on a smaller grid, resumably

Date: 2026-09-04

## Status

Accepted

Amends [4. Forecasts as a build artifact](0004-forecasts-as-a-build-artifact.md) and
[5. Truthful confidence intervals](0005-truthful-confidence-intervals.md).

## Context

ADR 0004 moved fitting off the request path and into `scripts/precompute_forecasts.py`, which
made the endpoint a sub-3ms lookup and made the batch the slow thing instead. Measured against
the real `data/names.built.db`: **4.30 s per name across 24,721 eligible pairs — 29.5 hours,
single-threaded.** A job that long is a job that gets run once and then never again, which is a
problem beyond patience: ADR 0005 makes the published interval labels a *product* of this batch,
so a batch nobody reruns is a set of coverage figures that silently stops describing the code
that produced them.

Profiling located the cost precisely. `fit_forecast` grid-searches up to 48 ARIMA models, and
`_validate` then grid-searches 48 more on the training slice — ~96 fits per name, at ~38 ms each,
and essentially nothing else (residual diagnostics are 0.002 s/name; the stationarity tests are
4.7 ms/name).

Four candidate savings were measured rather than assumed, because the ones that look best on a
stopwatch are not the ones that survive looking at their output:

| Change | Speed | Effect on the published forecast |
| --- | --- | --- |
| `multiprocessing` over names | 7.2x at 8 workers, **8.5x at 16** | none — names are independent |
| `concentrate_scale=True` | 1.79x | **median 8.86%**, p90 32.6% change |
| `enforce_stationarity/invertibility=False` | 1.39x | **median ~9%** change, and admits explosive AR roots |
| `MAX_P`/`MAX_Q` 3 -> 2 | ~1.8x | **median 0.00%**, p90 2.67% |

The statsmodels reparameterisations are fast and wrong for us: they change the selected order for
~75% of names and move the median name's five-year forecast by ~9%, which is a visibly different
chart on `/search`. Shrinking the grid is nearly free by comparison, and the order counts say why
— across a 60-name sample, **21 names selected ARIMA(0,1,0) and 16 selected (1,1,0)**; only 8
selected an order using p=3 or q=3, and the median AICc penalty for capping at 2 is 0.000. The
48-model grid mostly exists to rediscover that a random walk wins.

The `d` sweep turned out to be a correctness problem as well as a cost. `_fit_best_model` searched
`optimal_d - 1 .. optimal_d + 1` and picked the winner by AICc, but AICc is not comparable across
different differencing orders — differencing changes the data the likelihood is computed on. Those
three sets of models were never on a common scale, so the sweep was choosing between them on a
meaningless number.

Two further problems only appeared once the batch was actually run at scale rather than sampled:

- **A heavy tail, and no in-process way to bound it.** Every estimate above is a mean drawn from
  samples of 12-30 names, and the distribution is not summarised by its mean. Measured over 2,000
  real names with per-stage timing, the median name takes ~0.8 s and the 99th percentile ~3 s —
  but **0.1% of names never finish**, and roughly 25 such names exist in the full set. Unbounded,
  they are enough to occupy every worker indefinitely: a first attempt at the parallel batch ran
  53 minutes at 12 cores x 100% CPU and committed 500 forecasts.

  Two in-process attempts to cap them both failed, and the reasons generalise. Deriving the
  timeout from `Exception` let `_fit_best_model`'s `except Exception: continue` swallow it, after
  which the one-shot timer was spent and the name ran on unbounded. Fixing that to a
  `BaseException` still did nothing, because `signal.setitimer` alarms are only delivered between
  bytecode instructions and a runaway `ARIMA.fit()` stays inside compiled statsmodels/scipy code
  for minutes without returning to the interpreter. Capping the optimiser instead was measured and
  is a no-op: `maxiter=50` changed neither runtime nor a single forecast value across 700 names,
  because the fits that matter are not the ones exceeding 50 iterations.

- **Calibration was invocation-scoped.** Per-name coverage was accumulated in an in-memory dict
  and discarded, so `calibration` described the names processed by that one invocation. Any
  resume — the obvious response to a long job — would have calibrated on only the names it
  happened to refit, a non-random slice since rows are written in name order, and published a
  coverage figure that no longer described the population. ADR 0005 exists to stop exactly that
  class of untruth.

## Decision

**The batch fans out across cores, searches a smaller and better-founded grid, bounds each name,
and stores enough per name that its aggregates are properties of the table rather than of a run.**

- `_fit_best_model` searches only the differencing order the ADF/KPSS tests chose, and `MAX_P` and
  `MAX_Q` drop from 3 to 2: **96 fits per name become 18**, with the median forecast unchanged.
  `_validate` still runs its own grid search on the training slice rather than reusing the
  full-series order — reusing it would leak holdout information into the backtest and bias the
  measured coverage optimistically, which is the one direction that misleads.
- `run()` takes `workers`, **defaulting to 1**. The CLI defaults to `os.cpu_count()`. Thread
  environment variables are pinned to 1 before numpy is imported: N workers each spawning N BLAS
  threads oversubscribes badly enough to land slower than serial, and these series are far too
  small for threaded BLAS to pay for itself.
- `--timeout` (default 30 s) bounds one name, enforced by **`pebble.ProcessPool`, which kills and
  replaces the worker process**. Killing is not a stylistic preference: it is the only mechanism
  that stops compiled code, as the two failed in-process attempts above establish. On expiry no
  forecast is stored, which is the state `build_response` already renders as an empty forecast
  list, and the abandoned names are printed so a dropped name is distinguishable from an
  ineligible one.
- `run()` uses that same pool at every worker count, including `workers=1`. An earlier revision
  branched to a serial in-process path for `workers=1` so the test suite would avoid the pool;
  that meant the suite exercised code the real run never took, and left the one path without a
  working timeout. The equivalence between `workers=1` and `workers=2` is asserted directly
  instead.
- Each name's coverage contribution is stored in `forecasts.coverage_hits` / `coverage_n`, and
  `calibration` is recomputed as a sum over the whole table. `--resume` therefore skips stored
  names while still publishing a population figure. Coverage stays out of `payload`, which is
  served to the API verbatim.

## Consequences

The batch drops from ~29.5 hours to ~33 minutes at 12 workers (measured over 2,000 names with the
timeout in force, then scaled), which fits inside a lunch break on an 8-core desktop,
which is the difference between a figure that gets refreshed and one that rots. Forecast values
for the median name are unchanged; a minority shift slightly where the capped grid picks a
different order, and a small number of pathological names lose their forecast entirely to the
timeout and render as history-only — the same as any ineligible name.

`calibration` is now derivable from `forecasts` alone, so it can be recomputed without refitting.

`pebble` becomes a backend dependency. It is used only by this build-time script, and the cost is
accepted because the alternative is hand-rolling worker supervision -- the standard library's
pool cannot cancel a task already running inside compiled code.

About 25 names are expected to be abandoned to the timeout and to render as history-only. They are
named in the batch output rather than merely counted, so the set can be reviewed rather than
assumed benign.

Two things are deliberately left undone. Distributing across machines is not worth the credentials
and the data transfer for a job this size. And the order distribution strongly suggests most names
carry no forecastable signal beyond a random walk — `skill` from ADR 0005 already measures this per
name, so declining to forecast no-skill names is the cheapest optimisation left, but it is a
product decision about what `/search` shows, not a performance change, and is tracked separately.
