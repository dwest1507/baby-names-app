"""The pooled popularity forecaster: one model, every name.

Where the ARIMA pipeline fitted a model per name from that name's own history,
this fits one gradient-boosted model per forecast horizon across *every*
name's history at once, and predicts all of them in a single pass. A name's
forecast is then a function of what names shaped like it have gone on to do,
rather than of thirty points of its own. Six rounds of rolling-origin
research (`research/forecasting/`) measured the difference; see
docs/adr/0010-a-pooled-model-replaces-per-name-arima.md.

The configuration here is the one that research settled on, and it is
deliberately the plain one: the base feature block with no hand-built
interactions, no cohort or lifecycle features, popularity-weighted rows, one
seed, one booster per horizon, and the target expressed as log growth
relative to the origin year. `features` is a port of
`research/forecasting/pooled2.py`'s base block; `tests/test_forecast_pooled.py`
pins the two against a checked-in fixture so the port cannot drift from the
code that measured it.

Like `arima.py`, this is batch-only. It runs from
`scripts/precompute_forecasts.py`, is never imported by the application
package, and is not copied into the production image — so `lightgbm` is a
dev/build dependency and never reaches the container. See
docs/adr/0004-forecasts-as-a-build-artifact.md.
"""

import os

import numpy as np

from app.services.forecast import MIN_HISTORY_YEARS, is_eligible

# Five years ahead, one booster each.
H = 5

# Shares are strictly positive but arbitrarily small; this keeps log() finite
# for a share that rounds to zero rather than letting one name poison a fit.
FLOOR = 1e-9

# Boosting with row and column subsampling is a stochastic fit. Research
# averaged five seeds to keep one seed's noise out of a *measurement*; a
# published artifact instead needs the same input to produce the same bytes
# every time, so the batch pins one seed and reports it on the model card.
SEED = 0

# Training reaches back to name-origins from this year. Earlier SSA data is
# thin enough that the features built from it are mostly floor values.
FIRST_TRAIN_ORIGIN = 1930

# The base feature block: where the name's share is, how fast it has been
# moving over five different spans, whether that movement is accelerating,
# how noisy it is, and where it sits in its own history.
FEATURES = (
    "g1",
    "g2",
    "g3",
    "g5",
    "g10",
    "accel",
    "vol",
    "level",
    "level2",
    "below_peak",
    "yrs_since_peak",
    "age",
    "yrs_since_trough",
)

# Chosen in research on origins strictly earlier than any origin it was then
# scored on, and held fixed since. They are not re-tuned here: tuning against
# the artifact being published is how a build starts scoring itself.
HYPERPARAMETERS = {
    "num_leaves": 15,
    "learning_rate": 0.03,
    "n_estimators": 300,
    "min_child_samples": 200,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "max_bin": 255,
}

# The fit should care most about the names visitors actually look up, so rows
# are weighted by the name's share of births. The exponent softens that (a
# name a thousand times more popular counts thirty times more, not a thousand)
# and the clip stops a handful of giants from becoming the entire fit.
WEIGHT_POWER = 0.5
WEIGHT_CLIP = 50.0

# LightGBM's histogram building is deterministic only for a fixed thread
# count, so the batch pins one rather than taking whatever the machine
# happens to offer. Two runs at the same pinned count produce identical
# forecasts; see `tests/test_precompute_forecasts.py`.
THREADS = int(os.environ.get("FORECAST_THREADS") or (os.cpu_count() or 1))

# One pass over `names` in (name, sex, year) order, which is exactly the order
# `idx_names_name_sex_year` already stores (ADR 0009). SQLite walks the index
# and hands back rows already grouped and already sorted, so the extraction
# needs no temporary B-tree, no sort file, and no intermediate artifact on
# disk — `stream_series` closes each series as its run ends.
SERIES_SQL = (
    "SELECT name, sex, year, popularity_percent, popularity_rank "
    "FROM names ORDER BY name, sex, year"
)


def stream_series(conn):
    """Yield one `(key, years, values, rank)` per name/sex, streaming.

    `key` is `"name|sex"` with the name lowercased, matching how `forecasts`
    is keyed and how the history lookup normalises. `rank` is the name's rank
    in the last year it was observed, which for a name in current use is its
    rank in the newest year the database holds.

    Rows arrive grouped by the index, so a series is complete the moment the
    (name, sex) run ends and nothing more than the current run is ever held.
    """
    current_key = None
    years: list[int] = []
    values: list[float] = []
    rank = 0

    def finished():
        return (
            current_key,
            np.array(years, dtype=np.int32),
            np.array(values, dtype=np.float64),
            int(rank),
        )

    for name, sex, year, percent, name_rank in conn.execute(SERIES_SQL):
        key = f"{name.lower()}|{sex}"
        if key != current_key:
            if current_key is not None:
                yield finished()
            current_key, years, values = key, [], []
        years.append(year)
        values.append(percent)
        rank = name_rank or 0

    if current_key is not None:
        yield finished()


def _growth(log_series: np.ndarray, span: int) -> float:
    """Average log change per year over the last `span` years."""
    span = min(span, len(log_series) - 1)
    return (log_series[-1] - log_series[-1 - span]) / span if span >= 1 else 0.0


def features(log_train: np.ndarray) -> dict[str, float]:
    """The base feature block for one series, given its log share history.

    A port of `research/forecasting/pooled2.features` with the optional
    interaction and cohort blocks dropped, which is the configuration the
    research selected. Keeping it a pure function of one array is what lets
    the parity fixture pin it without a database.
    """
    n = len(log_train)
    steps = np.diff(log_train[-11:]) if n >= 3 else np.array([0.0])
    peak = int(np.argmax(log_train))
    trough = int(np.argmin(log_train[-30:])) if n >= 5 else 0
    g1, g2, g3, g5, g10 = (_growth(log_train, span) for span in (1, 2, 3, 5, 10))
    accel = g1 - g5
    level = log_train[-1]
    below_peak = log_train[-1] - log_train[peak]

    return {
        "g1": g1,
        "g2": g2,
        "g3": g3,
        "g5": g5,
        "g10": g10,
        "accel": accel,
        "vol": float(np.std(steps)) if len(steps) > 1 else 0.0,
        "level": level,
        # Scaled so the square does not dwarf every other column: log shares
        # run from about -14 to -3, so `level` alone is already a large number.
        "level2": (level / 10.0) ** 2,
        "below_peak": below_peak,
        "yrs_since_peak": min(n - 1 - peak, 60) / 60.0,
        "age": min(n, 145) / 145.0,
        "yrs_since_trough": min(len(log_train[-30:]) - 1 - trough, 30) / 30.0,
    }


def build_rows(series, origins) -> list[dict]:
    """One row per (series, origin) the series was eligible to be forecast at.

    Eligibility is `app.services.forecast.is_eligible` — the same rule the API
    explains to a visitor, applied at each origin rather than only at the
    newest year (ADR 0001). A row carries the features observable at the
    origin, the log growth actually realised over the next five years where it
    has been observed yet (`y`, NaN where it has not), and enough context to
    turn a prediction back into a share.
    """
    origins = sorted(origins)
    rows = []
    for key, years, values, rank in series:
        if len(years) < MIN_HISTORY_YEARS or years[-1] < origins[0]:
            continue
        future = {int(y): float(v) for y, v in zip(years, values, strict=True)}
        for origin in origins:
            observed = years <= origin
            if not is_eligible([int(y) for y in years[observed]], origin):
                continue
            log_train = np.log(np.maximum(values[observed], FLOOR))
            feature_row = features(log_train)
            target = np.full(H, np.nan)
            for step in range(1, H + 1):
                if (origin + step) in future:
                    target[step - 1] = np.log(max(future[origin + step], FLOOR)) - log_train[-1]
            rows.append(
                {
                    "key": key,
                    "rank": rank,
                    "origin": int(origin),
                    "x": np.array([feature_row[name] for name in FEATURES]),
                    "y": target,
                    "last": float(values[observed][-1]),
                    "level": float(log_train[-1]),
                    "actual": [future.get(origin + step) for step in range(1, H + 1)],
                }
            )
    return rows


def training_rows(series, origin: int) -> list[dict]:
    """Rows whose five-year outcome had already happened by `origin`.

    Nothing later than the origin being forecast may inform the fit, so the
    newest usable training origin is five years back: its targets closed in
    the origin year itself.
    """
    return build_rows(series, range(FIRST_TRAIN_ORIGIN, origin - H + 1))


def row_weights(rows) -> np.ndarray:
    """Popularity weights, normalised to mean 1 and clipped."""
    weights = np.exp(np.array([row["level"] for row in rows])) ** WEIGHT_POWER
    return np.clip(weights / weights.mean(), 0.0, WEIGHT_CLIP)


def train(rows, seed: int = SEED, threads: int = THREADS) -> list:
    """One booster per horizon, fitted on `rows`.

    Horizons are fitted independently rather than recursively: a five-year
    forecast is a direct prediction of year five, not year one applied five
    times, so an early error cannot compound down the path.
    """
    import lightgbm as lgb

    if not rows:
        raise ValueError("no training rows: the database has too little history to fit on")

    X = np.vstack([row["x"] for row in rows])
    Y = np.vstack([row["y"] for row in rows])
    weights = row_weights(rows)

    models = []
    for horizon in range(H):
        observed = ~np.isnan(Y[:, horizon])
        model = lgb.LGBMRegressor(
            objective="l2",
            random_state=seed,
            n_jobs=threads,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
            **HYPERPARAMETERS,
        )
        model.fit(X[observed], Y[observed, horizon], sample_weight=weights[observed])
        models.append(model)
    return models


def predict(models, rows) -> np.ndarray:
    """Forecast shares for every row at once: `(len(rows), H)`, all positive.

    The models predict log growth relative to the origin year, so a share is
    the origin's own share times the exponential of that. Nothing here can
    produce a negative value, which is why the pipeline carries no clamp at
    zero — the old one existed because ARIMA's additive intervals could reach
    below it.
    """
    if not rows:
        return np.zeros((0, H))
    X = np.vstack([row["x"] for row in rows])
    growth = np.column_stack([model.predict(X) for model in models])
    last = np.array([row["last"] for row in rows])
    return np.exp(growth) * last[:, None]


def log_residuals(rows, predicted: np.ndarray) -> np.ndarray:
    """`log(actual) - log(predicted)` for rows whose outcome is fully observed."""
    complete = [i for i, row in enumerate(rows) if all(v is not None for v in row["actual"])]
    if not complete:
        return np.zeros((0, H))
    actual = np.array([rows[i]["actual"] for i in complete], dtype=float)
    return np.log(np.maximum(actual, FLOOR)) - np.log(np.maximum(predicted[complete], FLOOR))


def band_offsets(residuals: np.ndarray, levels) -> dict[str, list[list[float]]]:
    """Two-sided log-residual quantiles per horizon, one pair per level.

    The band a forecast carries is the spread this model's own errors actually
    had, not a spread implied by a distributional assumption it never
    verified. Applying the offsets multiplicatively keeps both edges positive.
    """
    offsets = {}
    for level in levels:
        tail = (1 - level) / 2
        low = np.quantile(residuals, tail, axis=0)
        high = np.quantile(residuals, 1 - tail, axis=0)
        offsets[str(level)] = [[float(lo), float(hi)] for lo, hi in zip(low, high, strict=True)]
    return offsets


def apply_band(value: float, offsets: list[list[float]], horizon: int) -> tuple[float, float]:
    low, high = offsets[horizon]
    return value * float(np.exp(low)), value * float(np.exp(high))


def model_card(trained_through: int, training_rows_count: int, training_origins: int) -> dict:
    """What is true of the model itself, as opposed to of any one name.

    One model forecasts every name, so this is the same for all of them and is
    stored once rather than copied into 24,700 payloads. See
    `app.db_schema.CREATE_MODEL_CARD_TABLE`.
    """
    return {
        "model_name": "LightGBM Pooled Regressor (h=1..5)",
        "model_class": "gradient-boosted trees",
        "target": "log(y[t+h] / y[t])",
        "features": list(FEATURES),
        "horizons": H,
        "trained_through": int(trained_through),
        "training_origins": int(training_origins),
        "training_rows": int(training_rows_count),
        "sample_weight": f"share^{WEIGHT_POWER:g}",
        "seed": SEED,
    }
