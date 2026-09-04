# 4. Forecasts are a build artifact, not a request-time computation

Date: 2026-08-31

## Status

Accepted

## Context

Producing a forecast means grid-searching up to 48 ARIMA orders by AICc, twice — once to fit
the model that is actually returned, once more to refit on a holdout split for validation.
Measured end to end, a single name search cost 4.5-8.75 s, most of it CPU rather than I/O.

That cost sat on the request path, behind an endpoint with three properties that make CPU-bound
work there increasingly hard to defend:

- **No rate limit protected it** before ADR 0002's gateway landed, and even with the gateway in
  place, a visitor within their quota can still trigger a multi-second fit on every distinct name
  they search.
- **It will run behind a serverless function with a duration ceiling.** A slow fit risks a
  timeout rather than just a slow response, and the platform bills the CPU time whether or not
  the visitor waits for it.
- **The in-process result cache (`lru_cache` on `forecast_name`) is lost on every scale-to-zero
  sleep.** Under the deployment this app is moving to (ADR 0002; parent PRD dwest1507/baby-names-app#5),
  the container sleeps when idle, so the cache's warm state — the case the cache exists to make
  fast — is the rare one, and the cold, multi-second fit is the common one.

Separately, the source data is the SSA's annual release: it changes once a year, not once a
request. Fitting the same model on the same series repeatedly, once per visitor who searches
that name, computes the same answer every time.

## Decision

**Forecasts are computed offline by a batch script and stored, keyed on the lowercased name and
sex. The request path performs a lookup and fits nothing.**

- `backend/scripts/precompute_forecasts.py` reads the whole `names` table once and groups it in
  memory by `(LOWER(name), sex)`, rather than querying per name/sex pair. With ~24,721 eligible
  pairs in the real database (ADR 0001), one query per pair would pay the lookup cost that many
  times over for no benefit — the table fits comfortably in memory read once.
- For every pair eligible under ADR 0001's rule, the batch calls
  `app.services.forecast.fit_forecast` — the same function the API route used to call directly
  at request time — and writes the result into a new `forecasts` table
  (`backend/app/db_schema.py`, alongside the `names` table DDL so `build_db.py` and
  `make_sample_db.py` cannot let the two artifacts' shapes drift apart). Reusing the app's own
  fitting code, rather than a reimplementation in the batch, is the decisive correctness
  requirement here: a second implementation could silently diverge from what the live code would
  have produced, and there would be no test surface that would catch it.
- **The stored payload excludes the history series.** With history, a name's full forecast
  payload is roughly 8 KB; without it, roughly 1.2-1.7 KB. History is cheap to recompose at
  request time from an indexed lookup against `names` (ADR 0003), and storing it again in
  `forecasts` would re-add close to the gigabyte that pruning `names` to observed rows just
  removed. The endpoint composes its response from the stored blob plus history read fresh.
- `GET /api/names/{name}/forecast` (`backend/app/routes/names.py`) now does two lookups —
  `queries.get_name_history` and the new `queries.get_forecast` — and calls
  `forecast.build_response` to assemble them. Neither of those touches `statsmodels`. The route
  no longer wraps the call in `run_in_threadpool`: the previous wrapping existed specifically to
  keep the event loop free during CPU-bound fitting, and there is no CPU-bound work left to
  protect it from.
- The response shape is unchanged: `name`, `sex`, `history`, `forecast`, `validation`, `model`,
  with an empty `forecast` list and `null` `validation`/`model` for a name with no stored row —
  identical to how an ineligible name behaved before this change (`test_forecast` in
  `test_api.py` passes unmodified).
- **The batch has no test seam of its own.** The session-scoped `sample_db` fixture
  (`backend/tests/conftest.py`) runs `precompute_forecasts.run()` against the built sample
  database immediately after building it, so every existing API test exercises precomputed data
  automatically, the same way the deployed app will serve it. The batch's two properties that
  aren't observable through the HTTP surface — that it reads `names` once rather than per name,
  and that ineligible names get no stored row at all rather than an empty one — are asserted
  directly against the built artifact in `backend/tests/test_sample_db.py`; everything else
  (eligibility's effect on the response, response shape, that no fit happens live) is asserted
  through the existing HTTP test client, per the seam agreed in the parent PRD.
- The batch is invoked via a new Makefile target, `make precompute-forecasts`, run after
  `make build-db` — a backend script, not part of the data-pipeline notebooks, for the same
  reason as the build script: it has to import backend code, and the notebooks are a separate
  pipeline with their own dependency file that cannot cross that boundary.

## Consequences

- **The forecast endpoint responds in milliseconds**, not seconds: it is now two indexed lookups
  and no model fitting.
- **`statsmodels` is no longer imported on the request path in any live code path** — only
  `scripts/precompute_forecasts.py` calls `forecast.fit_forecast`, the function that reaches
  `_fit_best_model` and `ARIMA`. Verified in `test_api.py`
  (`test_forecast_endpoint_fits_no_model_at_request_time`) by monkeypatching
  `forecast._fit_best_model` and `forecast.fit_forecast` to raise, then asserting the endpoint
  still returns 200 with a populated forecast: if the route fit anything live, the patched
  function would raise and the request would 500 instead.
- **Running the batch is now a required step before deploying**, not an optional optimization.
  A freshly built database with no `forecasts` rows is not wrong — every name simply serves
  `forecast: []`, the same payload shape an ineligible name has always returned — but it is not
  what a visitor should see. This is a deploy-time discipline the Makefile and deployment
  documentation need to carry, not something the code can enforce.
- **Forecasts go stale between data refreshes**, by construction: the batch runs once per
  release of the underlying data, not once per request, so a stored forecast reflects whatever
  `names` looked like the last time `make precompute-forecasts` ran. This is the intended
  trade-off — the source data itself only updates annually, so a live fit would have recomputed
  the identical answer at 4.5-8.75 s of cost for no informational gain — but it means the batch
  must be re-run deliberately after every data refresh, not assumed to track it automatically.
- **The batch is expensive by design and must be run offline.** Grid-searching ARIMA orders for
  ~24,721 eligible pairs on the real database is a genuinely long-running job; it is a batch/CI
  or manual pre-deploy step, never something triggered inline from a request or a short-lived
  CI job with a tight timeout.
- `forecast.py` now exposes three functions where it previously exposed one
  (`forecast_name`, removed): `is_eligible` (the pure eligibility rule from ADR 0001, usable
  without a database connection), `fit_forecast` (the CPU-bound fit, batch-only), and
  `build_response` (the response composer, request-path-only, which fits nothing). Splitting
  these apart is what makes the "reuse the app's own code" requirement checkable at all — the
  batch and the route now provably run the same fitting function because there is only one.

## Related

- Parent PRD: dwest1507/baby-names-app#5
- Depends on: ADR 0001 (forecast eligibility rule), ADR 0003 (observed-rows-only database, the
  indexed history lookup this composes responses from)
- Implements: dwest1507/baby-names-app#9
