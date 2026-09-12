# Baby Names Explorer

A web application serving 145 years of Social Security Administration (SSA) baby name data with trend charts, pooled-model popularity forecasts, and a Groq-powered natural-language SQL chatbot.

## Language

**Observed Rows**:
Baby name records where at least 5 births were recorded for a given name, sex, and year in SSA records.
_Avoid_: Raw rows, zero-padded rows, fabricated entries

**Built Database**:
The deployable SQLite database artifact containing observed rows, canonical indexes, and precomputed forecast models.
_Avoid_: Source database, raw database, sample database

**Sample Database**:
A lightweight SQLite database with a subset of historical names used for fast local development and CI testing.
_Avoid_: Test database, stub database

**Precomputed Forecast**:
A 5-year statistical projection (point forecast and conformal uncertainty bands) computed offline for an eligible baby name.
_Avoid_: Dynamic forecast, live prediction

**Origin Year**:
The final observed calendar year in a training history from which a forecast or backtest step projects.
_Avoid_: Base year, cutoff year, anchor year, reference year

**Forecast Horizon**:
The multi-year window (currently 5 years) projected into unobserved time beyond an origin year.
_Avoid_: Forecast window, prediction period, forward span

**Validation Holdout**:
The most recent 5-year window of fully observed records used to evaluate forecast accuracy and residual calibration for display on the search page.
_Avoid_: Test set, evaluation slice, holdout window

**Backtest Span**:
The sequence of annual origin years (from 1995 through the latest year with a complete 5-year holdout) evaluated to measure per-name skill and conformal interval coverage.
_Avoid_: Benchmark sweep, rolling test origins, evaluation slice

**Popularity Tier**:
One of four rank-based frequency brackets (`top100`, `top1000`, `top5000`, `rest`) defined within an origin year, used to stratify evaluation, acceptance rules, and conformal interval calibration.
_Avoid_: Rank bracket, frequency bucket, volume tier

**Volatility Bin**:
One of three tertile brackets of a name's recent year-to-year log wobble, cut from the wobble present in the origin year being calibrated, used as the second axis of a conformal band's calibration stratum.
_Avoid_: Variance bucket, noise band, stability class

**Calibration Stratum**:
The `(popularity tier, volatility bin)` cell a name-origin falls into: the unit a conformal band's residual quantiles are estimated from, its achieved coverage is measured over, and the API reports back. A cell with too few observed outcomes borrows the whole population's band rather than estimating a tail of its own.
_Avoid_: Cohort, segment, calibration group

**Training Window**:
The bounded span of recent origin years (currently the most recent 40) a pooled fit may learn from; rows from earlier origins are discarded rather than down-weighted.
_Avoid_: Lookback, recency decay, training horizon

**Growth Cap**:
The per-horizon bound on a forecast's implied growth against its origin share, read off the largest five-year moves names actually made.
_Avoid_: Clamp, ceiling, outlier filter

**Path Smoothing**:
The endpoint-preserving moving average applied across a forecast's log steps so the five-year line reads as one trajectory rather than five independent per-horizon predictions.
_Avoid_: Curve fitting, interpolation, trend line

**Corpus Reconciliation**:
The single multiplicative factor applied per year, sex, and forecast horizon that scales every forecast in a slice onto the total share that sex held at the origin year.
_Avoid_: Normalization, rescaling, calibration

**Model Evaluation**:
The precomputed summary of backtest skill scores per popularity tier stored in the built database to self-certify artifact quality before deployment.
_Avoid_: Model metrics, benchmark report, score card


**Query Resource Budget**:
A wall-clock execution deadline enforced on model-generated SQL statements to prevent worker pool starvation from runaway queries.
_Avoid_: Query timeout, SQL gate


