"""Projected rank: what a projected share is ranked against.

A rank is a statement about a name's position among all the others, so the
field it is ranked against is the whole of the decision. Ranking the
forecastable names against each other is one sort and no extra data, and it is
wrong in a way that only shows up below the top 1000 — which is exactly where
ADR 0012's table puts a projected rank next to an actual one. See
docs/adr/0013-projected-rank-against-a-frozen-field.md.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.forecast import pooled  # noqa: E402


def test_a_frozen_competitor_holds_its_place_against_the_names_that_are_forecast():
    """The field is every name observed at the origin, not the eligible ones.

    Cleo cannot be forecast — too little history, or last recorded before the
    origin's newest year — so she is entered at the share she was last observed
    at rather than dropped. That is the same "carries on as it was" assumption
    the model is scored against, and it costs Bea a place she has not earned.
    Dropping Cleo instead asserts she vanishes, which is a stronger claim and
    always wrong in the same direction.
    """
    projected = {"ada|F": 0.03, "bea|F": 0.01}
    frozen = {"cleo|F": 0.02}

    ranks = pooled.rank_against_field(projected, frozen)

    assert ranks == {"ada|F": 1, "bea|F": 3}
    # The cheap alternative, and the reason it is not used: it flatters every
    # name with an un-forecastable competitor above it.
    assert pooled.rank_against_field(projected, {}) == {"ada|F": 1, "bea|F": 2}


def test_the_field_ranks_but_is_not_itself_ranked():
    """Only the names with a projection get a rank back."""
    ranks = pooled.rank_against_field({"ada|F": 0.01}, {"cleo|F": 0.02, "dot|F": 0.03})

    assert ranks == {"ada|F": 3}


def test_names_projected_to_the_same_share_share_the_better_rank():
    """Competition ranking, as `popularity_rank` itself is built.

    `build_db` ranks with `method="min"`, so a two-way tie for second is two
    seconds and no third. A projected rank printed beside an actual one has to
    resolve a tie the same way or the two columns are not comparable.
    """
    ranks = pooled.rank_against_field({"ada|F": 0.02, "bea|F": 0.02, "cleo|F": 0.01}, {})

    assert ranks == {"ada|F": 1, "bea|F": 1, "cleo|F": 3}


def _series(key: str, years, values):
    """One series shaped as `pooled.stream_series` yields them."""
    years = np.array(years, dtype=np.int32)
    return (key, years, np.array(values, dtype=float), np.zeros(len(years), dtype=np.int32))


def test_the_field_is_the_names_recorded_in_the_origin_year():
    """Recorded that year, at that year's share — not carried forward.

    An actual rank is a position among the names recorded in a year, so the
    field a projected rank is measured against is built the same way. A name
    that lapsed before the origin is out of the race, and one that had not
    arrived yet was never in it.
    """
    series = [
        _series("ada|F", range(2000, 2021), [0.01] * 20 + [0.03]),
        _series("bea|F", range(2000, 2015), [0.02] * 15),
        _series("cleo|F", range(2021, 2026), [0.05] * 5),
    ]

    assert pooled.observed_field(series, 2020) == {"ada|F": 0.03}


def test_a_projected_rank_is_a_position_among_names_of_the_same_sex():
    """`popularity_rank` is computed per (sex, year), so this has to be too.

    Every horizon is ranked separately as well: the field at horizon 1 is what
    the model said about the year after the origin, and at horizon 5 what it
    said about five years on.
    """
    rows = [{"key": "ada|F"}, {"key": "abe|M"}]
    # Ada is projected to fall over the five years and Abe to climb.
    predicted = np.array([[0.030, 0.025, 0.020, 0.015, 0.010], [0.001, 0.002, 0.003, 0.004, 0.005]])
    field = {"ada|F": 0.03, "abe|M": 0.001, "cleo|F": 0.02, "zeb|M": 0.004}

    ranks = pooled.projected_ranks(rows, predicted, field)

    # Ada is ranked against Cleo alone, Abe against Zeb alone: neither sex's
    # field can move the other's rank.
    assert list(ranks[0]) == [1, 1, 1, 2, 2]
    assert list(ranks[1]) == [2, 2, 2, 1, 1]


def test_the_field_a_name_is_ranked_in_holds_the_names_it_cannot_displace():
    """A forecastable name enters at its projection, not twice.

    Ada is in `rows`, so her frozen share is replaced rather than added: a name
    the model forecast is not also a name it froze, and counting her in both
    would have her beating herself.
    """
    rows = [{"key": "ada|F"}]
    predicted = np.array([[0.01] * pooled.H])
    field = {"ada|F": 0.05, "cleo|F": 0.02}

    assert list(pooled.projected_ranks(rows, predicted, field)[0]) == [2] * pooled.H
