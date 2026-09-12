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
seed, one booster per horizon, the target expressed as log growth relative to
the origin year, and a training pool bounded to the most recent `TRAIN_WINDOW`
origins. `features` is a port of `research/forecasting/pooled2.py`'s base
block; `tests/test_forecast_pooled.py` pins the two against a checked-in
fixture so the port cannot drift from the code that measured it.

What the boosters predict is not yet what the site draws. `point_forecasts`
applies three measured corrections on top, in an order that is not
interchangeable: the growth cap, then the path smoother, then reconciliation to
the corpus total. Each is a separate function above so it can be measured on
its own, and `tests/test_forecast_point_stack.py` pins what each one does.

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

# ...and no further back than this many origins, which in practice is the
# binding constraint. Research swept the window and found an interior optimum
# at four decades: training on *less* history costs skill, and so does
# training on more. Discarding the older rows outright beat discounting them
# by a factor of three on the tier below the top 100, which is not what a
# "older data is less relevant" story predicts — what the data supports is a
# bound on how much history the model wants, not a preference for recency.
# See research/forecasting/FINDINGS-6.md, section 2.
TRAIN_WINDOW = 40

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

# The growth cap reads its bound off the training outcomes rather than
# inventing one: all but a thousandth of the five-year moves names actually
# made. High enough that no ordinary forecast touches it, finite enough that a
# divergence cannot be drawn as a line. See `growth_caps`.
CAP_QUANTILE = 0.999

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


def training_origins(origin: int) -> range:
    """The origins a fit made at `origin` is allowed to learn from.

    Bounded at both ends. Nothing later than five years back, because a
    training row's five-year outcome has to have closed by the year being
    forecast from — otherwise the fit has seen the future. And nothing earlier
    than `TRAIN_WINDOW` origins before that, because the model does not want
    it.
    """
    newest = origin - H
    return range(max(FIRST_TRAIN_ORIGIN, newest - TRAIN_WINDOW + 1), newest + 1)


def training_rows(series, origin: int) -> list[dict]:
    """Rows whose five-year outcome had already happened by `origin`."""
    return build_rows(series, training_origins(origin))


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


def growth_caps(rows, quantile: float = CAP_QUANTILE) -> np.ndarray:
    """Per-horizon bound on |log(forecast / origin share)|, from observed moves.

    Only rows whose whole five-year window has been observed can say anything
    about how far a name moves in five years, so the others are left out
    rather than floored into the quantile.
    """
    observed = [row for row in rows if all(value is not None for value in row["actual"])]
    if not observed:
        return np.full(H, np.inf)
    actual = np.maximum(np.array([row["actual"] for row in observed], dtype=float), FLOOR)
    last = np.maximum(np.array([row["last"] for row in observed], dtype=float), FLOOR)
    return np.quantile(np.abs(np.log(actual / last[:, None])), quantile, axis=0)


def cap_growth(rows, predicted: np.ndarray, caps: np.ndarray) -> np.ndarray:
    """Clip each forecast's implied growth to `caps`, leaving the rest alone.

    A model fitted in log space can extrapolate multiplicative growth without
    limit; research measured a five-year ratio of 2.5e44 on nine name-origins
    out of ~13,000. Rare, and one of them is a chart a visitor can see is
    broken, so the guardrail stays. It is a clip and not a shrink: a forecast
    inside the bound comes back bit-for-bit unchanged.
    """
    if not rows:
        return predicted
    last = np.maximum(np.array([row["last"] for row in rows], dtype=float), FLOOR)
    growth = np.log(np.maximum(predicted, FLOOR) / last[:, None])
    clipped = np.clip(growth, -caps, caps)
    return np.where(clipped == growth, predicted, np.exp(clipped) * last[:, None])


def smooth_paths(rows, predicted: np.ndarray) -> np.ndarray:
    """Take the corners off each five-year path, without moving where it ends.

    One booster per horizon means nothing ties the five of them together: the
    model fitting year three has never seen year two's answer, so the path can
    rise, dip and rise again without that ever having been a claim about the
    name. A visitor reads the *shape* of the line, so the corners are noise
    presented as information.

    The fix is a three-point moving average over the path's log steps, padded
    at both edges with the end steps themselves. The padding is what makes it
    endpoint-preserving: each of the five raw steps enters the smoothed sum
    exactly three times, so the smoothed steps sum to the raw ones and the
    five-year value is untouched to floating point. Research measured it as an
    accuracy gain rather than a cosmetic one — reversals roughly halve, and
    poolSkill rises in the three tiers visitors look at (FINDINGS-6.md,
    section 3).
    """
    if not rows:
        return predicted
    last = np.array([row["last"] for row in rows], dtype=float)
    log_path = np.log(np.maximum(predicted, FLOOR))
    steps = np.diff(np.column_stack([np.log(np.maximum(last, FLOOR)), log_path]), axis=1)
    padded = np.column_stack([steps[:, :1], steps, steps[:, -1:]])
    kernel = np.ones(3) / 3.0
    smoothed = np.column_stack([padded[:, i : i + 3] @ kernel for i in range(steps.shape[1])])
    return np.exp(np.log(np.maximum(last, FLOOR))[:, None] + np.cumsum(smoothed, axis=1))


def reconcile(rows, predicted: np.ndarray) -> np.ndarray:
    """Scale each (sex, horizon) slice so the forecasts add up to what they must.

    One factor for everyone in the slice, applied multiplicatively. That is a
    constant shift in log space, so it leaves every name's rank and every
    ratio between two names exactly where the model put them: the constraint
    corrects the level of the whole cross-section, and expresses no opinion
    about any individual name.

    The alternatives were measured and are not used. Spreading the
    discrepancy equally in absolute terms takes far more from a small name
    than from a large one and has to be clipped at zero; spreading it by each
    name's own volatility loads it onto exactly the names whose futures are
    least certain. Neither is applied per tier either — a tier is a property
    of the evaluation, not of the adding-up constraint, and reconciling within
    tiers would make each one add up to a total nothing requires it to hit.
    """
    if not rows:
        return predicted
    reconciled = np.array(predicted, dtype=float)
    last = np.array([row["last"] for row in rows], dtype=float)
    # A slice is one origin's names of one sex: shares sum to a total within a
    # sex and within a year, and nothing is required to hold across either.
    slices = np.array([(row["origin"], row["key"].rsplit("|", 1)[1]) for row in rows])
    for origin, sex in np.unique(slices, axis=0):
        group = (slices[:, 0] == origin) & (slices[:, 1] == sex)
        totals = reconciled[group].sum(axis=0)
        reconciled[group] *= np.where(
            totals > 0, last[group].sum() / np.maximum(totals, FLOOR), 1.0
        )
    return reconciled


def point_forecasts(models, rows, caps: np.ndarray) -> np.ndarray:
    """The published point forecast: predict, cap, smooth, reconcile.

    The four steps are exposed separately above so each can be measured on its
    own, but they are only correct in this order and callers get them from
    here rather than composing them again. Smoothing preserves each path's
    five-year endpoint while moving years one through four, so it changes the
    sums reconciliation targets; reconciling first and smoothing afterwards
    would break the adding-up at four of the five horizons. The cap comes
    before both because it is a statement about the model's own output, and
    because a divergence left in would drag every other name's factor with it.
    """
    return reconcile(rows, smooth_paths(rows, cap_growth(rows, predict(models, rows), caps)))


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
