"""The shaded band, and the strata whose errors decide how wide it is.

A band is not derived from a distributional assumption the model never
checked. It is the spread this model's own five-year errors actually had, at
an origin early enough that the errors were observable — and it is that
spread taken *within a stratum*, because the errors of a top-100 name and the
errors of a name ranked 8,000 are not the same distribution, and neither are
those of a steady name and a jumpy one at the same rank.

A stratum is `(popularity tier, volatility bin)`. It is the unit the bands are
built from, the unit their achieved coverage is measured over, and the unit
the API reports back — so that "80% interval" on the chart means 80% for names
like this one, rather than 80% on average across a population whose tail is
badly wrong. See docs/adr/0011-conformal-bands-keyed-by-strata.md.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.forecast import pooled  # noqa: E402


def series_with_ranks(rank_by_year: dict[int, int], first_year: int = 1980) -> list:
    """One series whose rank changes over time, everything else held flat."""
    years = np.array(sorted(rank_by_year), dtype=np.int32)
    return [
        (
            "ada|F",
            years,
            np.full(len(years), 0.01),
            np.array([rank_by_year[int(year)] for year in years], dtype=np.int32),
        )
    ]


def test_a_row_carries_the_rank_the_name_held_at_its_own_origin():
    """A backtest row is tiered by what the name was *then*, not what it is now.

    Coverage measured at origin 2015 describes how the bands performed for
    names that were top-100 names in 2015. Tiering those rows by a rank from
    2025 would file a name that has since collapsed under `top100` and
    attribute its errors to a tier it was not in, which is how a tier's
    measured coverage stops describing the names it is served to.
    """
    series = series_with_ranks({year: year - 1970 for year in range(1980, 2026)})

    rows = {row["origin"]: row for row in pooled.build_rows(series, [1995, 2015, 2025])}

    assert rows[1995]["rank"] == 25
    assert rows[2015]["rank"] == 45
    assert rows[2025]["rank"] == 55


def test_an_unranked_year_does_not_pass_for_the_most_popular_name_there_is():
    """A missing rank must not read as rank 0, which would beat every name.

    `popularity_rank` is nullable in the source, and a null that becomes a
    zero lands in `top100` — the one tier whose coverage the deploy gate
    checks. Absent popularity is the opposite of extreme popularity.
    """
    series = series_with_ranks(dict.fromkeys(range(1980, 2026), 0))

    (row,) = pooled.build_rows(series, [2025])

    assert pooled.popularity_tier(row["rank"]) == "rest"


# Ranks that put a name in `top100`, `top1000` and `top5000` respectively.
# `rest` is deliberately never occupied, so there is always one stratum with
# no rows in it to check the fallback against.
RANK_CYCLE = (50, 500, 3000)


def wobbly_series(count: int = 630, last_year: int = 2020, first_year: int = 1940) -> list:
    """Series spanning a wide range of year-to-year wobble, from a fixed seed.

    The early ones barely move and the late ones lurch; `vol` is the standard
    deviation of the recent log steps, so this is the axis the volatility bins
    are meant to separate. Rank cycles independently of the wobble, so every
    (tier, bin) cell fills evenly rather than tier and bin standing in for
    each other.
    """
    rng = np.random.default_rng(4)
    years = np.arange(first_year, last_year + 1, dtype=np.int32)
    series = []
    for i in range(count):
        jitter = 0.01 + 0.35 * (i / count)
        noise = np.cumsum(rng.normal(0.0, jitter, len(years)))
        values = 10 ** rng.uniform(-5.0, -2.5) * np.exp(noise - noise.mean())
        series.append(
            (
                f"name{i:04d}|{'F' if i % 2 else 'M'}",
                years,
                np.maximum(values, 1e-7),
                np.full(len(years), RANK_CYCLE[i % len(RANK_CYCLE)], dtype=np.int32),
            )
        )
    return series


def test_the_volatility_bins_split_the_names_into_three_comparable_groups():
    """Tertiles of the wobble actually present, not thresholds chosen in advance.

    How jumpy a "jumpy" name is depends on the corpus — the edges have to come
    off the rows being calibrated, or a bin can come out empty on one database
    and hold everything on another. Three bins is what research conditioned
    on, and it is the coarsest split that still separates the steady names
    from the lurching ones.
    """
    rows = pooled.build_rows(wobbly_series(), [2020])

    edges = pooled.volatility_edges(rows)
    bins = [pooled.volatility_bin(row["vol"], edges) for row in rows]

    assert len(edges) == pooled.VOLATILITY_BINS - 1
    assert sorted(set(bins)) == list(range(pooled.VOLATILITY_BINS))
    counts = [bins.count(b) for b in range(pooled.VOLATILITY_BINS)]
    assert max(counts) - min(counts) <= 1


def test_a_steady_name_and_a_lurching_one_do_not_share_a_bin():
    """The bins are ordered by wobble, which is the whole point of having them."""
    rows = pooled.build_rows(wobbly_series(), [2020])
    edges = pooled.volatility_edges(rows)

    steady = min(rows, key=lambda row: row["vol"])
    lurching = max(rows, key=lambda row: row["vol"])

    assert pooled.volatility_bin(steady["vol"], edges) == 0
    assert pooled.volatility_bin(lurching["vol"], edges) == pooled.VOLATILITY_BINS - 1


@pytest.fixture(scope="module")
def calibrated() -> tuple[list[dict], np.ndarray, list[float], dict]:
    """A fit, its forecasts, and the bands built from the errors it made.

    One origin, trained and then forecast on the same series — which is what
    the batch's calibration origin does, except that there the outcome has
    since been observed. Here the series run to 2020 so the five-year window
    off origin 2015 is complete and the residuals are real.
    """
    series = wobbly_series()
    training = pooled.training_rows(series, 2015)
    models = pooled.train(training, threads=1)
    rows = pooled.build_rows(series, [2015])
    predicted = pooled.point_forecasts(models, rows, pooled.growth_caps(training))
    edges = pooled.volatility_edges(rows)
    return rows, predicted, edges, pooled.strata_bands(rows, predicted, (0.8, 0.95), edges)


def band_width(bands, level: str, stratum, horizon: int = 4) -> float:
    """How wide, in log space, the band a name in `stratum` would be given."""
    low, high = pooled.band_for(bands, level, stratum)[horizon]
    return high - low


def test_a_volatile_name_gets_a_wider_band_than_a_steady_one_at_the_same_rank(calibrated):
    """The band is about *this* name's uncertainty, not the average name's.

    A single population-wide band has one width for everybody, so it is
    necessarily too wide for the predictable names and too narrow for the
    unpredictable ones — and it is the second failure that matters, because it
    is the one that shows a visitor a confident band around a forecast nobody
    should be confident about. Conditioning on the volatility bin is what
    makes the shaded area mean something different for a lurching name.
    """
    rows, _, edges, bands = calibrated
    tier = max(
        pooled.TIERS,
        key=lambda t: sum(1 for row in rows if pooled.popularity_tier(row["rank"]) == t),
    )

    steady = band_width(bands, "0.8", (tier, 0))
    lurching = band_width(bands, "0.8", (tier, pooled.VOLATILITY_BINS - 1))

    assert lurching > steady * 1.5


def test_a_stratum_too_thin_to_measure_borrows_the_whole_populations_band(calibrated):
    """A quantile of six residuals is noise wearing a band's clothes.

    Conditioning only pays while each cell still has enough errors in it to
    estimate a tail from. A stratum below the threshold is not given a
    confidently wrong band of its own; it gets the global one, which is the
    honest fallback and keeps every name banded.
    """
    _, _, _, bands = calibrated
    # No series is ranked past 5000, so this cell holds no residuals at all.
    absent = ("rest", 0)

    assert absent not in bands["0.8"]
    assert pooled.band_for(bands, "0.8", absent) == bands["0.8"][pooled.GLOBAL_STRATUM]


def test_every_stratum_the_bands_do_carry_was_measured_on_enough_rows(calibrated):
    """A populated stratum is one that cleared the threshold, not one that appeared."""
    rows, predicted, edges, bands = calibrated
    counted: dict[tuple[str, int], int] = {}
    for row in rows:
        if all(value is not None for value in row["actual"]):
            key = (pooled.popularity_tier(row["rank"]), pooled.volatility_bin(row["vol"], edges))
            counted[key] = counted.get(key, 0) + 1

    strata = set(bands["0.8"]) - {pooled.GLOBAL_STRATUM}

    assert strata
    assert strata == {key for key, n in counted.items() if n >= pooled.MIN_STRATUM_ROWS}


def test_the_wider_band_is_the_wider_one_at_both_published_levels(calibrated):
    """A stratum that is harder to predict is harder to predict at 95% too."""
    rows, _, _, bands = calibrated

    for level in ("0.8", "0.95"):
        strata = sorted(set(bands[level]) - {pooled.GLOBAL_STRATUM})
        assert strata
        for stratum in strata:
            assert band_width(bands, level, stratum) > 0
        widest = max(strata, key=lambda s: band_width(bands, level, s))
        assert widest[1] == pooled.VOLATILITY_BINS - 1


def test_the_95_band_contains_the_80_band_in_every_stratum(calibrated):
    """Nesting is not optional: the chart draws one inside the other."""
    _, _, _, bands = calibrated

    for stratum in bands["0.8"]:
        wide = pooled.band_for(bands, "0.95", stratum)
        narrow = pooled.band_for(bands, "0.8", stratum)
        for horizon in range(pooled.H):
            assert wide[horizon][0] <= narrow[horizon][0]
            assert narrow[horizon][1] <= wide[horizon][1]


def test_the_stratum_a_name_is_reported_is_the_one_whose_band_it_was_given(calibrated):
    """Reported coverage has to describe the band actually handed out.

    A name in a stratum too thin to earn its own band gets the global one. If
    its holdout were still counted into its own cell, that cell would publish
    a coverage figure for a band nobody in it was ever given — and publish it
    from a handful of points, which is the precise defect ADR 0005 refused to
    paper over. So the cell a name's outcome is counted into is the cell whose
    band it used.
    """
    _, _, _, bands = calibrated
    measured = next(iter(set(bands["0.8"]) - {pooled.GLOBAL_STRATUM}))

    assert pooled.band_stratum(bands, "0.8", measured) == measured
    assert pooled.band_stratum(bands, "0.8", ("rest", 0)) == pooled.GLOBAL_STRATUM


def test_calibrating_on_an_origin_with_no_observed_outcome_says_so():
    """There is nothing to calibrate on, and that has to be sayable.

    Every origin the batch calibrates at is old enough that its five-year
    window has closed, so this does not arise there. It arises the moment
    anything else asks for bands at a recent origin — and `pooled.train`
    already refuses a fit with no rows the same way, rather than letting an
    `IndexError` out of a quantile over zero residuals stand as the
    explanation.
    """
    series = wobbly_series(count=12, last_year=2025)
    rows = pooled.build_rows(series, [2025])

    with pytest.raises(ValueError, match="no observed outcomes"):
        pooled.strata_bands(rows, np.ones((len(rows), pooled.H)), (0.8,), [0.1, 0.2])
