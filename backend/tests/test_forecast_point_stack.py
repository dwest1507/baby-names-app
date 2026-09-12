"""The three refinements that sit between the booster and the published line.

`pooled.predict` turns the boosters' log-growth predictions back into shares,
and on its own that is not yet the forecast the site publishes. Research
measured three corrections on top of it, and this is where their *behaviour*
is pinned — the numbers they produce are pinned separately, against the
research harness itself, in `test_forecast_pooled.py`.

* a bounded training window, because four decades of origins beat nine,
* a smoothed path, because one booster per horizon has nothing tying the five
  of them into a trajectory,
* reconciliation, because shares within a sex sum to a fixed total and
  forecasting names one at a time is free to violate that.

See docs/adr/0010-a-pooled-model-replaces-per-name-arima.md.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.forecast import pooled  # noqa: E402


def make_series(count: int = 40, first_year: int = 1900, last_year: int = 2025) -> list:
    """Hump-shaped series, enough of them to fit on, from a fixed seed."""
    rng = np.random.default_rng(11)
    series = []
    for i in range(count):
        years = np.arange(first_year, last_year + 1, dtype=np.int32)
        peak = rng.integers(first_year + 10, last_year)
        shape = np.exp(-(((years - peak) / rng.uniform(8.0, 40.0)) ** 2))
        noise = np.cumsum(rng.normal(0.0, 0.05, len(years)))
        values = 10 ** rng.uniform(-5.0, -2.0) * shape * np.exp(noise - noise.mean())
        series.append(
            (
                f"name{i:03d}|{'F' if i % 2 else 'M'}",
                years,
                np.maximum(values, 1e-7),
                np.full(len(years), rng.integers(1, 9000), dtype=np.int32),
            )
        )
    return series


@pytest.fixture(scope="module")
def series() -> list:
    return make_series()


def test_training_reaches_back_a_bounded_window_of_origins(series):
    """Four decades of origins, not nine.

    The pool used to start at 1930 and weight a 1935 name-origin exactly like
    a 2014 one. Research swept the window and found an interior optimum —
    both less history and more of it cost skill — so the cutoff is part of the
    model, not a performance concession.
    """
    origins = sorted({row["origin"] for row in pooled.training_rows(series, 2025)})

    assert len(origins) == pooled.TRAIN_WINDOW
    # Nothing later than five years before the origin: a training row's
    # five-year outcome has to have closed by the year being forecast from.
    assert origins[-1] == 2025 - pooled.H
    assert origins[0] == 2025 - pooled.H - pooled.TRAIN_WINDOW + 1


def paths(*rows_of_values) -> tuple[list[dict], np.ndarray]:
    """Hand-written five-year paths, each starting from a share of 1.0."""
    rows = [{"key": f"n{i}|F", "origin": 2025, "last": 1.0} for i in range(len(rows_of_values))]
    return rows, np.array(rows_of_values, dtype=float)


def reversals(last: float, path) -> int:
    """Direction changes along a path, the jaggedness a visitor actually sees."""
    steps = np.diff(np.log(np.concatenate([[last], np.asarray(path, dtype=float)])))
    signs = np.sign(np.where(np.abs(steps) < 0.02, 0, steps))
    nonzero = signs[signs != 0]
    return int((np.diff(nonzero) != 0).sum()) if len(nonzero) > 1 else 0


def test_smoothing_leaves_each_path_where_it_ended(series):
    """The five-year number does not move, and that is the point.

    The smoother is a three-point moving average over the path's log steps
    with edge padding, and the padding makes the smoothed steps sum to exactly
    the raw ones. So it redistributes the shape of the line and cannot change
    where the line arrives — which is what lets it ship as an accuracy change
    to years one through four rather than as a new five-year forecast.
    """
    rows, raw = paths(
        [1.0, 1.4, 1.2, 1.9, 1.5],
        [0.9, 0.8, 0.7, 0.6, 0.5],
        [2.0, 2.0, 2.0, 2.0, 2.0],
    )

    smoothed = pooled.smooth_paths(rows, raw)

    np.testing.assert_allclose(smoothed[:, -1], raw[:, -1], rtol=1e-12)


def test_smoothing_removes_direction_changes_the_model_never_claimed(series):
    """A path that zigzags is five guesses in a row, not a trajectory."""
    rows, raw = paths([1.3, 0.9, 1.3, 0.9, 1.3])

    smoothed = pooled.smooth_paths(rows, raw)

    assert reversals(1.0, raw[0]) == 4
    assert reversals(1.0, smoothed[0]) < reversals(1.0, raw[0])


def test_smoothing_keeps_a_genuine_bend(series):
    """It is a moving average, not a straight line.

    Research tried the straight line too: it is the smoothest path available,
    and it is the worst arm in both tail tiers, because a tail name's five
    years mostly consist of flattening out and a line in log space cannot
    express that. The bend carries information; only the corner does not.
    """
    rows, raw = paths([1.5, 1.9, 2.1, 2.2, 2.25])

    smoothed = pooled.smooth_paths(rows, raw)[0]
    steps = np.diff(np.log(np.concatenate([[1.0], smoothed])))

    assert np.all(steps > 0)
    assert steps[0] > steps[-1] * 2


def test_reconciled_forecasts_sum_to_the_share_the_origin_actually_held(series):
    """Shares add up, so the forecasts have to add up too.

    `popularity_percent` is a share of one year's births within a sex, so
    across every name of a sex it sums to a fixed total. Every forecast here
    is made one name at a time and nothing ties them together, so the sum of
    them is free to drift — and it does, upward, which means the site would
    otherwise predict growth for more names than can possibly grow. A per-name
    error metric cannot see that at all.

    The total to hit is the one observed at the origin: over five years the
    eligible set's total moves by at most about a percent, so "it stays where
    it was" is both the honest forecast of it and an easy one.
    """
    rows = [
        {"key": "ada|F", "origin": 2025, "last": 0.6},
        {"key": "bea|F", "origin": 2025, "last": 0.4},
        {"key": "cal|M", "origin": 2025, "last": 1.5},
    ]
    raw = np.array(
        [
            [0.7, 0.8, 0.9, 1.0, 1.1],
            [0.5, 0.5, 0.5, 0.5, 0.5],
            [1.8, 2.0, 2.2, 2.4, 2.6],
        ]
    )

    reconciled = pooled.reconcile(rows, raw)

    np.testing.assert_allclose(reconciled[:2].sum(axis=0), 1.0, rtol=1e-12)
    np.testing.assert_allclose(reconciled[2], 1.5, rtol=1e-12)


def test_reconciliation_moves_every_name_in_a_slice_by_the_same_factor(series):
    """Multiplicative and global: one factor per year, sex and horizon.

    Every name in the slice is multiplied by it, so the ratio between any two
    forecasts survives untouched and nothing is pushed toward zero. An
    additive spread would do neither.
    """
    rows = [{"key": f"n{i}|F", "origin": 2025, "last": 0.1 * (i + 1)} for i in range(6)]
    raw = np.array([[0.1 * (i + 1) * (1.0 + 0.04 * h) for h in range(5)] for i in range(6)])

    reconciled = pooled.reconcile(rows, raw)

    ratios = reconciled / raw
    for horizon in range(5):
        np.testing.assert_allclose(ratios[:, horizon], ratios[0, horizon], rtol=1e-12)
    assert np.all(reconciled > 0)


def test_reconciliation_reads_no_popularity_tier(series):
    """There is no per-tier path, so rank cannot change the answer."""
    raw = np.array([[0.2, 0.21, 0.22, 0.23, 0.24], [0.8, 0.79, 0.78, 0.77, 0.76]])
    popular = [
        {"key": "ada|F", "origin": 2025, "last": 0.2, "rank": 3},
        {"key": "bea|F", "origin": 2025, "last": 0.8, "rank": 9},
    ]
    obscure = [
        {"key": "ada|F", "origin": 2025, "last": 0.2, "rank": 19_000},
        {"key": "bea|F", "origin": 2025, "last": 0.8, "rank": 21_000},
    ]

    np.testing.assert_array_equal(pooled.reconcile(popular, raw), pooled.reconcile(obscure, raw))


def test_a_runaway_projection_is_clipped_to_what_names_have_actually_done(series):
    """The guardrail against a divergence being drawn as a forecast.

    A model working in log space can extrapolate multiplicative growth without
    limit, and research saw exactly that: a five-year ratio of 2.5e44 on a
    handful of name-origins. It is rare, and one of them is a visibly broken
    chart. The bound is read off the data rather than invented — the largest
    five-year move any name actually made, at the origins the model trained
    on. Beyond that is not a forecast.
    """
    training = pooled.training_rows(series, 2025)
    caps = pooled.growth_caps(training)
    rows = [
        {"key": "ada|F", "origin": 2025, "last": 1.0},
        {"key": "bea|F", "origin": 2025, "last": 1.0},
    ]
    absurd = np.array([[1e9, 1e12, 1e15, 1e18, 1e21], [1e-9, 1e-12, 1e-15, 1e-18, 1e-21]])

    capped = pooled.cap_growth(rows, absurd, caps)

    assert np.all(caps > 0)
    np.testing.assert_allclose(capped[0], np.exp(caps), rtol=1e-12)
    np.testing.assert_allclose(capped[1], np.exp(-caps), rtol=1e-12)


def test_an_ordinary_forecast_passes_the_cap_untouched(series):
    """A guardrail that bites on ordinary forecasts is a model, not a guard."""
    training = pooled.training_rows(series, 2025)
    caps = pooled.growth_caps(training)
    rows = [{"key": "ada|F", "origin": 2025, "last": 0.5}]
    ordinary = np.array([[0.52, 0.55, 0.57, 0.6, 0.62]])

    np.testing.assert_array_equal(pooled.cap_growth(rows, ordinary, caps), ordinary)


def roughness(rows, predicted: np.ndarray) -> float:
    """Mean absolute second difference of the log paths — how cornered they are."""
    last = np.array([row["last"] for row in rows], dtype=float)
    steps = np.diff(np.column_stack([np.log(last), np.log(predicted)]), axis=1)
    return float(np.abs(np.diff(steps, axis=1)).mean())


def test_the_point_stack_smooths_the_path_and_then_makes_it_add_up(series):
    """The order is the whole content of this test, and it is not reversible.

    Smoothing preserves each path's five-year endpoint but moves years one
    through four, so it changes the very sums reconciliation exists to fix.
    Run the other way round, the adding-up that reconciliation just imposed
    would be broken again by the smoother, at four of the five horizons.

    So both properties can only hold at once in one order: the published paths
    are smoother than the boosters' raw output *and* they sum, at every
    horizon, to the share their sex actually held at the origin.
    """
    training = pooled.training_rows(series, 2025)
    models = pooled.train(training, threads=1)
    rows = pooled.build_rows(series, [2025])
    caps = pooled.growth_caps(training)

    raw = pooled.predict(models, rows)
    published = pooled.point_forecasts(models, rows, caps)

    assert roughness(rows, published) < roughness(rows, raw)
    for sex in ("F", "M"):
        group = [i for i, row in enumerate(rows) if row["key"].endswith(f"|{sex}")]
        target = sum(rows[i]["last"] for i in group)
        np.testing.assert_allclose(published[group].sum(axis=0), target, rtol=1e-12)


def test_each_origin_year_gets_its_own_factor(series):
    """A slice is one year of one sex, because that is what has to sum to a total.

    Shares add up within a sex within a year, and nothing requires anything to
    hold across two years. Pooling two origins into one factor would impose a
    constraint on their combined total that the data never asserts.
    """
    rows = [
        {"key": "ada|F", "origin": 2025, "last": 1.0},
        {"key": "bea|F", "origin": 2020, "last": 4.0},
    ]
    raw = np.array([[2.0] * 5, [2.0] * 5])

    reconciled = pooled.reconcile(rows, raw)

    np.testing.assert_allclose(reconciled[0], 1.0, rtol=1e-12)
    np.testing.assert_allclose(reconciled[1], 4.0, rtol=1e-12)
