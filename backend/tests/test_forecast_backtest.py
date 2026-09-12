"""The rolling-origin backtest: the span, the window it trains on, the tally.

A forecast the search page labels with "this name beats no-change by 12%" is
making a claim about the name, and one holdout window cannot support it — the
2021-25 window alone is mostly a measurement of the 2020 birth-rate shock.
These cover the machinery that measures a name across every window it was
eligible for instead. See docs/adr/0010-a-pooled-model-replaces-per-name-arima.md.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.forecast import pooled  # noqa: E402


def test_the_span_reaches_every_origin_whose_five_year_window_has_closed():
    """The last origin a backtest can score is five years before the newest.

    The span is read off the data rather than written down: a database
    updated to 2025 scores 1995-2020, and next year's, unchanged, scores
    1995-2021. Hardcoding the end is how a 2026 rebuild would quietly go on
    reporting a score it measured against a window it could no longer see.
    """
    span = pooled.backtest_span(2025)

    assert list(span) == list(range(1995, 2021))
    assert len(span) == 26
    assert list(pooled.backtest_span(2026)) == list(range(1995, 2022))


def flat_series(count: int, first_year: int = 1940, last_year: int = 2025) -> list:
    """`count` name/sex series, each observed every year of the same span."""
    years = np.arange(first_year, last_year + 1, dtype=np.int32)
    return [
        (
            f"name{i}|F",
            years,
            np.full(len(years), 0.01 * (i + 1)),
            np.full(len(years), i + 1, dtype=np.int32),
        )
        for i in range(count)
    ]


def test_the_window_builds_each_origin_once_and_lets_the_old_ones_go(monkeypatch):
    """Feature rows are built once for the whole batch, and held only while wanted.

    Twenty-seven fits over a forty-origin training window is 1,080
    origin-passes if each fit builds its own rows, and feature extraction —
    not the boosting — is what the batch spends its time on. Advancing one
    window across the span instead builds each origin exactly once.

    The other half is the reason it is a *window* and not a cache: rows fall
    out of the far end as the span advances, so backtesting 26 origins holds
    no more memory than fitting one does. Keeping them all would grow the
    batch's footprint by the length of the span, which is how it gets
    OOM-killed in CI.
    """
    series = flat_series(3)
    built: list[int] = []
    real = pooled.build_rows
    monkeypatch.setattr(
        pooled, "build_rows", lambda s, origins: (built.extend(origins), real(s, origins))[1]
    )

    window = pooled.TrainingWindow(series)
    span = list(pooled.backtest_span(2025)) + [2025]
    held = []
    for origin in span:
        training, rows = window.advance(origin)
        assert training and rows
        assert {row["origin"] for row in rows} == {origin}
        assert {row["origin"] for row in training} == set(pooled.training_origins(origin))
        held.append(len(window.held_origins))

    assert len(built) == len(set(built)), "an origin was built more than once"
    assert max(held) <= pooled.TRAIN_WINDOW + pooled.H


def scored_row(key: str, origin: int, rank: int = 1, actual: float = 2.0, last: float = 1.0):
    """The fields the tally reads off a backtest row, and nothing else."""
    return {
        "key": key,
        "rank": rank,
        "vol": 0.0,
        "origin": origin,
        "x": np.zeros(len(pooled.FEATURES)),
        "y": np.zeros(pooled.H),
        "last": last,
        "level": np.log(last),
        "actual": [actual] * pooled.H,
    }


def test_a_names_skill_is_the_average_of_every_window_it_was_eligible_for():
    """One window is a measurement of one five-year period, not of the name.

    The 2021-25 holdout contains the birth-rate shock; a name scored on that
    alone is labelled with how the pandemic went, and next year's rebuild
    would relabel it with something else. Averaging over every window since
    1995 is what makes the figure on the chart a property of the name.
    """
    tally = pooled.BacktestTally()
    # Halves the naive error at the first origin, matches it at the second:
    # a name that is averagely predictable, measured twice.
    tally.add(1995, [scored_row("ada|F", 1995)], np.full((1, pooled.H), 1.5))
    tally.add(2000, [scored_row("ada|F", 2000)], np.full((1, pooled.H), 1.0))

    assert tally.skill_per_name()["ada|F"] == {"skill": pytest.approx(0.25), "skill_windows": 2}


def test_a_name_eligible_at_fewer_origins_is_scored_on_the_windows_it_has():
    """Short histories are averaged over less, not dropped.

    A name first recorded in 2010 has five windows to the 1995 name's
    twenty-six. Requiring the full span would leave every recent name — which
    is most of what visitors search — with no skill figure at all, and
    scoring it against windows it was not eligible for would credit the model
    with forecasts it never made.
    """
    tally = pooled.BacktestTally()
    tally.add(1995, [scored_row("ada|F", 1995)], np.full((1, pooled.H), 1.5))
    tally.add(
        2000,
        [scored_row("ada|F", 2000), scored_row("zoe|F", 2000)],
        np.full((2, pooled.H), 2.0),
    )

    per_name = tally.skill_per_name()

    assert per_name["zoe|F"] == {"skill": pytest.approx(1.0), "skill_windows": 1}
    assert per_name["ada|F"]["skill_windows"] == 2


def test_a_window_whose_outcome_is_not_fully_observed_is_not_a_window():
    """A name suppressed partway through a window has no outcome to score.

    The source drops a name/year below five births, so `actual` can be short
    at any origin. Scoring the years that did survive would compare a
    five-year forecast against a three-year one; the honest count is the
    windows that closed.
    """
    incomplete = scored_row("ada|F", 1995)
    incomplete["actual"] = [2.0, 2.0, None, 2.0, 2.0]

    tally = pooled.BacktestTally()
    tally.add(1995, [incomplete], np.full((1, pooled.H), 1.5))
    tally.add(2000, [scored_row("ada|F", 2000)], np.full((1, pooled.H), 1.5))

    assert tally.skill_per_name()["ada|F"] == {"skill": pytest.approx(0.5), "skill_windows": 1}


def test_a_tiers_score_weights_the_errors_that_are_large():
    """`pool_skill` and `med_skill` answer different questions, and both ship.

    Pooling sums the errors before dividing, so a tier's score is dominated by
    the names whose forecasts are most wrong in absolute terms — which is what
    an acceptance rule should be measuring, since those are the charts a
    visitor most notices being wrong. The median of the per-window skills is
    the counterweight: a model that is excellent on the giants and useless on
    everything else scores well on one and badly on the other, and a deploy
    gate reading only the first would never see it.
    """
    tally = pooled.BacktestTally()
    tally.add(
        1995,
        [
            scored_row("small|F", 1995, rank=5),
            scored_row("large|F", 1995, rank=50, actual=200.0, last=100.0),
        ],
        np.array([[1.5] * pooled.H, [200.0] * pooled.H]),
    )

    top100 = tally.evaluation()["top100"]

    # 2.5 of 505 summed absolute error, against a median of the two windows'
    # 0.5 and 1.0.
    assert top100["pool_skill"] == pytest.approx(1 - 2.5 / 505)
    assert top100["med_skill"] == pytest.approx(0.75)


def test_the_evaluation_records_how_many_origins_stand_behind_each_tier():
    """A score is only as good as the span it was measured over.

    `verify-db` refuses a build whose backtest skipped origins, so the count
    has to be the artifact's own rather than an assumption about how the batch
    was invoked — a run cut short must publish 12 origins and be caught, not
    publish 26 and pass.
    """
    tally = pooled.BacktestTally()
    for origin in (1995, 2000, 2005):
        tally.add(origin, [scored_row("ada|F", origin, rank=5)], np.full((1, pooled.H), 1.5))
    tally.add(2005, [scored_row("zoe|F", 2005, rank=900)], np.full((1, pooled.H), 1.5))

    evaluation = tally.evaluation()

    assert evaluation["top100"]["origins_evaluated"] == 3
    assert (evaluation["top100"]["min_origin"], evaluation["top100"]["max_origin"]) == (1995, 2005)
    assert evaluation["top1000"]["origins_evaluated"] == 1
    # A tier the span never populated is absent, not zero: "measured, and bad"
    # and "never measured" must not read the same to a deploy gate.
    assert "top5000" not in evaluation


def test_the_window_trains_on_exactly_what_a_standalone_build_would():
    """The window is an optimisation, and must not be a change of model.

    Row order is not cosmetic here: the fit subsamples rows and columns, so
    handing LightGBM the same rows in a different order fits a different
    model. Building the window origin-by-origin would naturally produce them
    grouped by origin, where `training_rows` produces them grouped by name —
    and `tests/test_forecast_pooled.py` pins the research parity of the
    latter. If the batch trained on the other one, the fixture would no longer
    certify the forecasts the batch publishes.
    """
    series = flat_series(4)
    window = pooled.TrainingWindow(series)

    for origin in (2015, 2016, 2025):
        training, rows = window.advance(origin)
        expected = pooled.training_rows(series, origin)
        assert [(row["key"], row["origin"]) for row in training] == [
            (row["key"], row["origin"]) for row in expected
        ]
        assert [row["key"] for row in rows] == [
            row["key"] for row in pooled.build_rows(series, [origin])
        ]
