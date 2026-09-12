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

import heapq
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

# The four rank brackets every evaluation, acceptance rule and band
# calibration is broken down on (CONTEXT.md, "Popularity Tier"). Research has
# reported every number this way since round 1, because the tiers behave
# differently enough that a population average hides the tail: the ARIMA
# pipeline these replaced scored +0.161 on the top 100 and -0.413 beyond rank
# 5000.
TIERS = ("top100", "top1000", "top5000", "rest")
TIER_BOUNDS = ((100, "top100"), (1000, "top1000"), (5000, "top5000"))

# The second axis of a calibration stratum. Two names at the same rank can
# have very different futures: research found a band conditioned on tier alone
# still under-covers the jumpy names and over-covers the steady ones, and
# three bins is the coarsest split that separates them. The edges are tertiles
# of the wobble actually present in the rows being calibrated, not fixed
# thresholds — "volatile" is only meaningful relative to the corpus, and a
# hardcoded cut can leave a bin empty on one database and holding everything
# on another. See research/forecasting/conformal.py.
VOLATILITY_BINS = 3

# A stratum needs enough rows for a quantile of its residuals to mean
# anything; below this it borrows the whole population's instead. Research's
# threshold, unchanged.
MIN_STRATUM_ROWS = 60

# The stratum standing for "every name", used when a cell has too few
# residuals of its own to estimate a tail from. Not a tier and not a bin, so
# it cannot collide with a real stratum.
GLOBAL_STRATUM = ("*", -1)

# The growth cap reads its bound off the training outcomes rather than
# inventing one: all but a thousandth of the five-year moves names actually
# made. High enough that no ordinary forecast touches it, finite enough that a
# divergence cannot be drawn as a line. See `growth_caps`.
CAP_QUANTILE = 0.999

# The first origin the rolling backtest scores. Earlier SSA years are thin
# enough — and far enough from how names move now — that a skill figure
# averaged over them would describe a different corpus from the one being
# forecast. The end of the span is not written down at all: it is wherever the
# data's newest five-year window closes. See CONTEXT.md, "Backtest Span".
FIRST_BACKTEST_ORIGIN = 1995


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
    """Yield one `(key, years, values, ranks)` per name/sex, streaming.

    `key` is `"name|sex"` with the name lowercased, matching how `forecasts`
    is keyed and how the history lookup normalises. `ranks` runs alongside
    `years`: the name's rank in each year it was observed, so a row built at
    an origin can be tiered by the rank the name held *then*. A tier is a
    property of an origin year rather than of the name (CONTEXT.md,
    "Popularity Tier"), and a backtest that tiered 1995 rows by a 2025 rank
    would credit each tier with the errors of a different set of names.

    Rows arrive grouped by the index, so a series is complete the moment the
    (name, sex) run ends and nothing more than the current run is ever held.
    """
    current_key = None
    years: list[int] = []
    values: list[float] = []
    ranks: list[int] = []

    def finished():
        return (
            current_key,
            np.array(years, dtype=np.int32),
            np.array(values, dtype=np.float64),
            np.array(ranks, dtype=np.int32),
        )

    for name, sex, year, percent, name_rank in conn.execute(SERIES_SQL):
        key = f"{name.lower()}|{sex}"
        if key != current_key:
            if current_key is not None:
                yield finished()
            current_key, years, values, ranks = key, [], [], []
        years.append(year)
        values.append(percent)
        # `popularity_rank` is nullable in the source; 0 carries "no rank
        # recorded" onward, and `popularity_tier` is the one place that says
        # what that means.
        ranks.append(name_rank or 0)

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


def popularity_tier(rank: int) -> str:
    """Which rank bracket a name sat in, at the origin the rank was read from.

    A rank of zero or less means the source recorded no rank for that year.
    That is an absence of measured popularity, not an extreme of it, so it
    falls to the bottom bracket — where a bare `rank <= 100` comparison would
    instead file it under `top100`, the one tier the deploy gate checks.
    """
    if rank <= 0:
        return TIERS[-1]
    for bound, tier in TIER_BOUNDS:
        if rank <= bound:
            return tier
    return TIERS[-1]


def volatility_edges(rows) -> list[float]:
    """The tertile cuts of `vol` across `rows`, as `VOLATILITY_BINS - 1` edges.

    Read once off the origin the bands are calibrated at and then reused at
    every other origin, so a name's bin means the same thing whether it is
    being calibrated, scored or served.
    """
    quantiles = np.arange(1, VOLATILITY_BINS) / VOLATILITY_BINS
    vols = np.array([row["vol"] for row in rows], dtype=float)
    return (
        [float(edge) for edge in np.quantile(vols, quantiles)]
        if len(vols)
        else [0.0] * (VOLATILITY_BINS - 1)
    )


def volatility_bin(vol: float, edges) -> int:
    """Which volatility bin a wobble of `vol` falls in: 0 is the steadiest."""
    return int(np.searchsorted(np.asarray(edges, dtype=float), float(vol)))


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
    for key, years, values, ranks in series:
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
                    # The rank and the wobble as they stood at this origin:
                    # together they are the row's calibration stratum, and
                    # both have to be read at the origin rather than at the
                    # end of the series. `vol` is the feature block's own
                    # volatility, lifted out so the strata cannot come to
                    # disagree with what the model reads.
                    "rank": int(ranks[observed][-1]),
                    "vol": feature_row["vol"],
                    "origin": int(origin),
                    "x": np.array([feature_row[name] for name in FEATURES]),
                    "y": target,
                    "last": float(values[observed][-1]),
                    "level": float(log_train[-1]),
                    "actual": [future.get(origin + step) for step in range(1, H + 1)],
                }
            )
    return rows


def backtest_span(max_observed_year: int) -> range:
    """The origins whose five-year outcome the database can already check.

    An origin belongs in the span only if every year it forecasts has since
    been observed, so the newest one it can hold is `H` years back. On the
    2025 database that is 1995 through 2020 — 26 windows — and on next year's,
    with nothing here changed, 1995 through 2021.
    """
    return range(FIRST_BACKTEST_ORIGIN, max_observed_year - H + 1)


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


class TrainingWindow:
    """One advancing window of feature rows, shared by every fit in a batch.

    The batch fits at 27 origins — 26 backtest origins and the production one
    — and each fit trains on the `TRAIN_WINDOW` origins behind it. Built
    independently that is over a thousand passes of feature extraction, which
    is the batch's dominant cost and almost all of it redundant: consecutive
    origins want training windows that overlap in thirty-nine of forty years.

    So the rows are built once and advanced. `advance` adds whatever origins
    the new fit needs, and releases everything that has fallen off the back of
    the window — which is what keeps the footprint of a 26-origin backtest the
    same as a single fit's, rather than the whole span's. What it holds is the
    training window plus the handful of newer origins not yet in it: each of
    those was forecast at, and will be trained on `H` fits later, so releasing
    it would only mean building it twice.

    It is an optimisation and nothing more: what comes out is the same rows in
    the same order `training_rows` would have built them in. That is not
    cosmetic. The fit subsamples rows, so the same rows in a different order
    fit a different model, and it is `training_rows` that the research parity
    fixture pins. Stored per origin but handed back grouped by name, which is
    what the k-way merge below is for: each origin's rows are already in the
    corpus's name order, so merging them on that key reassembles the original
    ordering without a sort.
    """

    def __init__(self, series):
        self._series = series
        self._rows: dict[int, list[dict]] = {}
        self._position = {key: index for index, (key, *_) in enumerate(series)}

    @property
    def held_origins(self) -> set[int]:
        """The origins whose rows are in memory right now."""
        return set(self._rows)

    def advance(self, origin: int) -> tuple[list[dict], list[dict]]:
        """Move to `origin`: the rows to train on, and the rows to forecast.

        Origins must be visited in increasing order — the window only ever
        moves forward, and an origin it has already released is gone.
        """
        wanted = training_origins(origin)
        for year in list(self._rows):
            if year < wanted.start:
                del self._rows[year]
        for year in list(wanted) + [origin]:
            if year not in self._rows:
                self._rows[year] = build_rows(self._series, [year])
        # Ties go to the earliest iterable, so passing the origins in
        # ascending order puts one name's rows in origin order — exactly what
        # `build_rows` produces from a sorted list of origins.
        training = list(
            heapq.merge(
                *(self._rows[year] for year in wanted),
                key=lambda row: self._position[row["key"]],
            )
        )
        return training, self._rows[origin]


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


def observed_rows(rows) -> list[int]:
    """Indices of the rows whose whole five-year outcome has been observed.

    The caps, the residuals and the band strata all have to agree about which
    rows have an outcome, so they all ask here rather than each re-deriving
    it.
    """
    return [i for i, row in enumerate(rows) if all(v is not None for v in row["actual"])]


def window_skills(rows, predicted: np.ndarray):
    """`(key, tier, skill)` for each row whose five-year outcome is observed.

    `skill` is the standard comparison against the naive forecast — the
    origin's own share held flat for five years: `1 - model MAE / naive MAE`,
    so 0 is "no better than assuming nothing changes" and negative is worse.
    It is computed over the whole five-year path rather than per horizon,
    which is the unit the search page reports and the unit research scored on.

    The tier is the one the name held *at this origin*, not now, so a name
    that has since collapsed contributes its errors to the tier it was
    actually in when the forecast was made.
    """
    for i in observed_rows(rows):
        row = rows[i]
        actual = np.array(row["actual"], dtype=float)
        model_error = float(np.abs(actual - predicted[i]).sum())
        naive_error = float(np.abs(actual - row["last"]).sum())
        yield (
            row["key"],
            popularity_tier(row["rank"]),
            1 - model_error / naive_error if naive_error > 0 else 0.0,
            model_error,
            naive_error,
        )


class BacktestTally:
    """What the rolling backtest measured, accumulated one origin at a time.

    Two different questions are being asked of the same windows, and they need
    different arithmetic. A *name* gets the plain average of its own window
    skills, because each window is one equally-valid measurement of how
    predictable that name is. A *tier* gets the ratio of summed errors
    (`pool_skill`), because a tier's score should be dominated by the names
    whose errors are large rather than by the many small ones — alongside the
    median window skill (`med_skill`), which is the opposite view and is
    reported next to it for exactly that reason.

    Only the running sums are kept, not the windows, so the tally costs the
    same whether the span is one origin or twenty-six.
    """

    def __init__(self):
        self._names: dict[str, list[float]] = {}
        self._tier_errors: dict[str, np.ndarray] = {tier: np.zeros(2) for tier in TIERS}
        self._tier_skills: dict[str, list[float]] = {tier: [] for tier in TIERS}
        self._tier_origins: dict[str, set[int]] = {tier: set() for tier in TIERS}

    def add(self, origin: int, rows, predicted: np.ndarray) -> None:
        """Score one origin's forecasts against what actually happened."""
        for key, tier, skill, model_error, naive_error in window_skills(rows, predicted):
            total, count = self._names.get(key, (0.0, 0))
            self._names[key] = (total + skill, count + 1)
            self._tier_errors[tier] += (model_error, naive_error)
            self._tier_skills[tier].append(skill)
            self._tier_origins[tier].add(origin)

    def skill_per_name(self) -> dict[str, dict]:
        """Each name's average skill, and how many windows it rests on.

        The window count travels with the figure because it is what qualifies
        it: a name eligible since 1995 has been measured 26 times and a name
        first recorded in 2015 has been measured once, and the two numbers do
        not deserve the same confidence.
        """
        return {
            key: {"skill": total / count, "skill_windows": count}
            for key, (total, count) in self._names.items()
        }

    def evaluation(self) -> dict[str, dict]:
        """Per-tier scores, for the artifact to certify itself with.

        One row per tier that the span actually populated. A tier nothing was
        scored in is left out rather than published as zero — a deploy gate
        reading this must be able to tell "measured, and bad" from "never
        measured". See `app.db_schema.CREATE_MODEL_EVALUATION_TABLE`.
        """
        evaluation = {}
        for tier in TIERS:
            origins = self._tier_origins[tier]
            if not origins:
                continue
            model_error, naive_error = self._tier_errors[tier]
            evaluation[tier] = {
                "pool_skill": float(1 - model_error / naive_error) if naive_error > 0 else 0.0,
                "med_skill": float(np.median(self._tier_skills[tier])),
                "origins_evaluated": len(origins),
                "min_origin": min(origins),
                "max_origin": max(origins),
            }
        return evaluation


def growth_caps(rows, quantile: float = CAP_QUANTILE) -> np.ndarray:
    """Per-horizon bound on |log(forecast / origin share)|, from observed moves.

    Only rows whose whole five-year window has been observed can say anything
    about how far a name moves in five years, so the others are left out
    rather than floored into the quantile.
    """
    observed = [rows[i] for i in observed_rows(rows)]
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
    """`log(actual) - log(predicted)`, one row per fully observed outcome.

    Aligned with `observed_rows(rows)`, not with `rows`: a row whose five-year
    window ran off the end of its series has no error to report.
    """
    complete = observed_rows(rows)
    if not complete:
        return np.zeros((0, H))
    actual = np.array([rows[i]["actual"] for i in complete], dtype=float)
    return np.log(np.maximum(actual, FLOOR)) - np.log(np.maximum(predicted[complete], FLOOR))


def band_offsets(residuals: np.ndarray, level: float) -> list[list[float]]:
    """Two-sided log-residual quantiles per horizon, as `[low, high]` pairs.

    The band a forecast carries is the spread this model's own errors actually
    had, not a spread implied by a distributional assumption it never
    verified. Applying the offsets multiplicatively keeps both edges positive.
    """
    tail = (1 - level) / 2
    low = np.quantile(residuals, tail, axis=0)
    high = np.quantile(residuals, 1 - tail, axis=0)
    return [[float(lo), float(hi)] for lo, hi in zip(low, high, strict=True)]


def row_stratum(row, edges) -> tuple[str, int]:
    """The calibration stratum a row belongs to: `(tier, volatility bin)`.

    Both halves are read at the row's own origin, so the same function serves
    the historical backtest (tiering a 2015 row by its 2015 rank) and serving
    (tiering a production row by its rank in the newest observed year).
    """
    return (popularity_tier(row["rank"]), volatility_bin(row["vol"], edges))


def strata_bands(rows, predicted: np.ndarray, levels, edges, min_rows=MIN_STRATUM_ROWS) -> dict:
    """Band offsets per `(level, stratum)`, from the errors this fit made.

    Returns `{level_str: {stratum: offsets}}`, always including
    `GLOBAL_STRATUM`. A stratum appears in its own right only once it has
    `min_rows` observed outcomes behind it; `band_for` falls back to the
    global band for the rest, because a tail estimated from six residuals is
    a confidently wrong band rather than a conditioned one.

    Conditioning is the whole point: a single population band has one width
    for every name, so it is necessarily too wide for the predictable ones
    and too narrow for the rest — and the second failure is the one that
    matters, since it draws a confident band around a forecast nobody should
    be confident about. See
    docs/adr/0011-conformal-bands-keyed-by-strata.md.
    """
    residuals = log_residuals(rows, predicted)
    if not len(residuals):
        # A band is a measurement of errors that happened. At an origin whose
        # five-year window has not closed there are none, and the honest
        # answer is to say so rather than to publish a band of zero width.
        raise ValueError(
            f"no observed outcomes to calibrate bands from at "
            f"origin {rows[0]['origin'] if rows else '?'}: the five-year window "
            "has not closed yet"
        )
    groups: dict[tuple[str, int], list[int]] = {}
    for position, index in enumerate(observed_rows(rows)):
        groups.setdefault(row_stratum(rows[index], edges), []).append(position)

    bands: dict[str, dict[tuple[str, int], list[list[float]]]] = {}
    for level in levels:
        bands[str(level)] = {GLOBAL_STRATUM: band_offsets(residuals, level)}
        for stratum, positions in groups.items():
            if len(positions) >= min_rows:
                bands[str(level)][stratum] = band_offsets(residuals[positions], level)
    return bands


def band_stratum(bands: dict, level, stratum) -> tuple[str, int]:
    """The stratum whose band a name in `stratum` actually receives.

    Itself where it earned one, `GLOBAL_STRATUM` where it did not. Coverage is
    counted into *this* cell rather than into the name's own, so a published
    figure always describes the band the names behind it were given — and so a
    thin cell cannot publish a coverage estimate from a handful of points,
    which is the defect ADR 0005 refused to paper over.
    """
    stratum = tuple(stratum)
    return stratum if stratum in bands[str(level)] else GLOBAL_STRATUM


def band_for(bands: dict, level, stratum) -> list[list[float]]:
    """The offsets a name in `stratum` gets, global where it has none of its own."""
    at_level = bands[str(level)]
    return at_level[band_stratum(bands, level, stratum)]


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
