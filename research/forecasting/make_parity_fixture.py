"""Pin the pooled pipeline's numbers, so the port cannot drift from the research.

`backend/scripts/forecast/pooled.py` is a tidied copy of what this harness
measured: `pooled2.features`' base block, `pooled2.rows_for`' row builder, and
`pooled3.train_gbt`/`forecast_gbt`. Copies drift. This script runs a set of
series through the harness modules themselves and writes what *they* produced
into `backend/tests/fixtures/pooled_parity.json`;
`backend/tests/test_forecast_pooled.py` then replays the same series through
the shipped code and demands the same answers.

The series are generated here rather than extracted from `data/names.built.db`
on purpose. A fixture has to run in CI, where the 1.1 GB database is not
available and never will be, and it has to be small enough that nobody
minds it being in git. These are shaped like SSA series — a rise, a peak, a
decline, autocorrelated noise, shares spanning five orders of magnitude — and
they are drawn from a fixed seed, so the fixture is regenerable rather than
magic.

The check this fixture enables is a numeric one, so both sides have to be
pinned to one thread: LightGBM's histogram construction is only bit-for-bit
reproducible at a fixed thread count, and `pooled3` asks OpenMP for every core
it can see. Regenerate deliberately, and only when the model is meant to
change:

    cd research/forecasting && ../../backend/.venv/bin/python3 make_parity_fixture.py
"""

import json
import os
import sys

# Before numpy or lightgbm is imported anywhere: `pooled3` fits with
# `n_jobs=-1`, which resolves to whatever OpenMP reports, and the whole point
# of the fixture is a number that does not depend on the machine that made it.
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_var] = "1"

import numpy as np  # noqa: E402

SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SP)
import pooled2  # noqa: E402
import pooled3  # noqa: E402

REPO = os.path.dirname(os.path.dirname(SP))
OUT = os.path.join(REPO, "backend", "tests", "fixtures", "pooled_parity.json")

ORIGIN = 2025
SERIES_COUNT = 250
SEED = 0
THREADS = 1

# The shipped configuration: base features only, one seed, popularity-weighted
# rows. Mirrors `backend/scripts/forecast/pooled.HYPERPARAMETERS`.
HP = {"leaves": 15, "lr": 0.03, "trees": 300, "min_child": 200}
SETS: set[str] = set()


def generate_series(rng):
    """Series shaped like the real ones, drawn from a fixed seed.

    A name is a hump: it climbs to a peak, falls away from it, and wobbles
    around that path year to year. Spreading the peaks, widths, levels and
    start years over the ranges the real data covers is what makes the
    training set varied enough for the model to actually split on a feature
    rather than collapse to a mean.
    """
    series = []
    for i in range(SERIES_COUNT):
        start = int(rng.integers(1910, 1996))
        end = ORIGIN if i % 20 else int(rng.integers(ORIGIN - 20, ORIGIN))
        years = np.arange(start, end + 1, dtype=np.int32)
        if len(years) < 15:
            continue

        peak_year = int(rng.integers(start + 2, end + 1))
        spread = float(rng.uniform(6.0, 45.0))
        peak_share = float(10 ** rng.uniform(-5.5, -1.9))

        shape = np.exp(-(((years - peak_year) / spread) ** 2))
        # Autocorrelated log noise: year-to-year wobble that persists, which
        # is what makes `vol` and `accel` carry information at all.
        noise = np.cumsum(rng.normal(0.0, 0.06, len(years)))
        noise -= noise.mean()
        values = peak_share * shape * np.exp(noise)
        values = np.maximum(values, 1e-7)

        series.append(
            (
                f"name{i:04d}|{'F' if i % 2 else 'M'}",
                years,
                np.round(values, 9),
                int(rng.integers(1, 12000)),
            )
        )
    return series


def main() -> None:
    rng = np.random.default_rng(7)
    series = generate_series(rng)

    training = pooled2.train_rows(series, ORIGIN, SETS, None)
    rows = pooled2.rows_for(series, [ORIGIN], SETS, None)
    models, _ = pooled3.train_gbt(
        series,
        ORIGIN,
        SETS,
        None,
        HP,
        weight="pop",
        power=0.5,
        clip=50.0,
        extra=None,
        objective="l2",
        seed=SEED,
    )
    predicted = np.vstack(pooled3.forecast_gbt(models, rows))

    payload = {
        "generated_by": "research/forecasting/make_parity_fixture.py",
        "origin": ORIGIN,
        "seed": SEED,
        "threads": THREADS,
        "hyperparameters": HP,
        "feature_names": pooled2.feat_names(SETS),
        "training_rows": len(training),
        "series": [
            {
                "key": str(key),
                "rank": int(rank),
                "years": [int(y) for y in years],
                "values": [round(float(v), 9) for v in values],
            }
            for key, years, values, rank in series
        ],
        "features": [
            {"key": row["key"], "x": [float(v) for v in row["x"]]} for row in rows
        ],
        "predicted": [[float(v) for v in row] for row in predicted],
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    print(
        f"{len(series)} series, {len(training):,} training rows, "
        f"{len(rows)} forecast -> {OUT} ({os.path.getsize(OUT) / 1024:.0f} KB)"
    )


if __name__ == "__main__":
    main()
