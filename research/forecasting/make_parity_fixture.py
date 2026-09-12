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
import cap  # noqa: E402
import pooled2  # noqa: E402
import pooled3  # noqa: E402
import reconcile  # noqa: E402
import smooth  # noqa: E402

REPO = os.path.dirname(os.path.dirname(SP))
OUT = os.path.join(REPO, "backend", "tests", "fixtures", "pooled_parity.json")

ORIGIN = 2025
SERIES_COUNT = 250
SEED = 0
THREADS = 1

# The point stack rounds 5 and 6 added on top of the booster, mirroring
# `backend/scripts/forecast/pooled`: train on the most recent 40 origins, clip
# the implied growth at the 99.9th percentile of what names actually did,
# smooth the path with a three-point moving average over its log steps, and
# scale each (sex, horizon) slice onto the share that sex held at the origin.
WINDOW = 40
CAP_QUANTILE = 0.999

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


def apply_point_stack(training, rows, predicted):
    """Cap, smooth and reconcile, through the harness modules that measured them.

    Returns the caps alongside the published paths. The caps are pinned
    separately because a bound at the 99.9th percentile of what names actually
    do is one no ordinary forecast reaches — nothing in this fixture is
    clipped — so the published numbers alone would not notice the cap being
    derived wrongly, or at all.

    Two small departures from calling those modules blind, both forced:

    * `cap.caps_from` reads `r["actual"]` straight, which is a NaN for a row
      whose five-year window ran off the end of its series. It was only ever
      fed forecast files, where those rows are already gone; here the
      incomplete ones are dropped first.
    * `reconcile.Panel` indexes years against `arange(1880, 2025)`, so it
      cannot hold origin 2025 at all. The target it would compute is the
      eligible set's summed share at the origin, and every row's `last` *is*
      that name's share at the origin (`rows_for` drops any name not observed
      there), so the sum of `last` is the same number.
    """
    complete = [r for r in training if all(v is not None for v in r["actual"])]
    caps = cap.caps_from(complete, {r["origin"] for r in complete}, CAP_QUANTILE)

    out = []
    for row, path in zip(rows, predicted, strict=True):
        last = max(row["last"], cap.FLOOR)
        clipped = np.clip(np.log(np.maximum(path, cap.FLOOR) / last), -caps, caps)
        out.append(smooth.smooth_ma(np.exp(clipped) * last, row["last"]))
    out = np.vstack(out)

    for sex in sorted({str(r["key"]).rsplit("|", 1)[1] for r in rows}):
        group = [i for i, r in enumerate(rows) if str(r["key"]).rsplit("|", 1)[1] == sex]
        target = float(sum(rows[i]["last"] for i in group))
        for h in range(out.shape[1]):
            out[group, h] = reconcile.reconcile_group(out[group, h], None, target, "prop")
    return caps, out


def main() -> None:
    rng = np.random.default_rng(7)
    series = generate_series(rng)

    training = pooled2.train_rows(series, ORIGIN, SETS, None, window=WINDOW)
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
        window=WINDOW,
    )
    predicted = np.vstack(pooled3.forecast_gbt(models, rows))
    caps, published = apply_point_stack(training, rows, predicted)

    payload = {
        "generated_by": "research/forecasting/make_parity_fixture.py",
        "origin": ORIGIN,
        "seed": SEED,
        "threads": THREADS,
        "hyperparameters": HP,
        "feature_names": pooled2.feat_names(SETS),
        "training_rows": len(training),
        "training_origins": sorted({int(r["origin"]) for r in training}),
        "window": WINDOW,
        "cap_quantile": CAP_QUANTILE,
        "caps": [float(v) for v in caps],
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
        "published": [[float(v) for v in row] for row in published],
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
