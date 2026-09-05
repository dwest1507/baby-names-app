"""Where the benchmark's inputs live, and how its evaluation sample is drawn."""

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
BACKEND = os.path.join(REPO, "backend")
WORK = os.environ.get("FORECAST_WORK", os.path.join(HERE, ".work"))
SERIES = os.path.join(WORK, "series.npz")
DB = os.environ.get("NAMES_DB_PATH", os.path.join(REPO, "data", "names.built.db"))


def bucket(rank):
    """Popularity tier by 2024 rank — the axis every table is broken down on."""
    return (
        "top100"
        if rank <= 100
        else "top1000"
        if rank <= 1000
        else "top5000"
        if rank <= 5000
        else "rest"
    )


def load_all():
    """Every extracted series, as (key, years, values, 2024 rank) tuples.

    The arrays are hoisted out of the NpzFile first: indexing it decompresses
    the whole member on every access, which turns this into minutes.
    """
    d = np.load(SERIES, allow_pickle=True)
    keys, years, vals, rank = d["keys"], d["years"], d["vals"], d["rank"]
    return [(keys[i], years[i], vals[i], int(rank[i])) for i in range(len(keys))]


def load(sample_top=None, sample_mid=None, sample_rest=None, seed=0):
    """A popularity-stratified evaluation sample.

    Popular names are what visitors look at, so the default is to take *every*
    top-1000 name and sample the tiers below it — the tail is there to prove a
    method does not fall over on it, not to be measured exhaustively.
    """
    d = np.load(SERIES, allow_pickle=True)
    keys, years, vals, rank = d["keys"], d["years"], d["vals"], d["rank"]
    rng = np.random.default_rng(seed)

    def pick(idx, n):
        return idx if n is None or n >= len(idx) else rng.choice(idx, n, replace=False)

    sel = np.concatenate(
        [
            pick(np.where(rank <= 1000)[0], sample_top),
            pick(np.where((rank > 1000) & (rank <= 5000))[0], sample_mid),
            pick(np.where(rank > 5000)[0], sample_rest),
        ]
    )
    return [(keys[i], years[i], vals[i], int(rank[i])) for i in sel]
