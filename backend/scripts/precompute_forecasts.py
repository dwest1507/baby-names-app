"""Precompute forecasts for every eligible name/sex and store them.

Forecasting runs once, offline, and writes its result into the `forecasts`
table so that the API becomes a lookup and the request path fits nothing. See
docs/adr/0004-forecasts-as-a-build-artifact.md.

What runs here is the pooled model (`scripts/forecast/pooled.py`): one
gradient-boosted booster per horizon, trained across every name's history at
once and predicting all ~24,700 eligible names in a single pass. That shape is
why this script is much smaller than the one it replaces. Fitting a separate
ARIMA per name meant 24,700 independent CPU-bound grid searches, which in turn
meant a worker pool to get through them, a per-name timeout for the handful
that never converged, and a resume flag so a crash did not cost the whole run.
One fit for all names needs none of those, and they are gone; see
docs/adr/0010-a-pooled-model-replaces-per-name-arima.md.

Each fit is followed by the point stack — the growth cap, the path smoother
and reconciliation to the corpus total, in that order — so what is calibrated,
scored and stored are all the same forecasts the search page draws. See
`pooled.point_forecasts`.

The batch trains at three origins, all of them in the past relative to what
they are used for:

* `production` — the newest observed year. Its forecast is what gets served.
* `holdout` — five years earlier, so its five-year forecast can be scored
  against years that have since been observed. This is the per-name validation
  the search page shows.
* `calibration` — five years earlier again, so the spread of *its* errors can
  set the published bands without those bands having seen the holdout they are
  then measured on. Coverage in the `calibration` table is therefore an
  out-of-sample measurement, which is the whole point of
  docs/adr/0005-truthful-confidence-intervals.md.

Usage: uv run python scripts/precompute_forecasts.py [db_path] [--threads N]
"""

import os

# LightGBM is deterministic for a fixed thread count, so the batch pins one
# rather than inheriting whatever the machine offers. This has to happen
# before numpy is imported anywhere, hence before the imports below.
_THREADS = os.environ.get("FORECAST_THREADS") or str(os.cpu_count() or 1)
os.environ.setdefault("FORECAST_THREADS", _THREADS)
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, _THREADS)

import argparse  # noqa: E402
import json  # noqa: E402
import sqlite3  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent))

from app import db_schema  # noqa: E402
from scripts.forecast import pooled  # noqa: E402

REPO_ROOT = Path(__file__).parent.parent.parent
DEFAULT_DB = str(REPO_ROOT / "data" / "names.built.db")

# The nominal levels the API publishes bands for. What each one actually
# covers is measured, not assumed, and written to `calibration`.
LEVELS = (0.8, 0.95)


def _ensure_schema(conn) -> None:
    """Create the batch's tables, widening an older `forecasts` if needed.

    `CREATE TABLE IF NOT EXISTS` is a no-op against a `forecasts` table built
    before coverage was stored per name, so an artifact from an earlier run
    would keep its three columns and fail the insert.
    """
    conn.execute(db_schema.CREATE_FORECASTS_TABLE)
    conn.execute(db_schema.CREATE_CALIBRATION_TABLE)
    conn.execute(db_schema.CREATE_MODEL_CARD_TABLE)
    existing = {row[1] for row in conn.execute("PRAGMA table_info(forecasts)")}
    for column in ("coverage_hits", "coverage_n"):
        if column not in existing:
            conn.execute(f"ALTER TABLE forecasts ADD COLUMN {column} TEXT")


def _fit(series, origin: int, note, threads: int):
    """Train at one origin and forecast every name eligible there.

    What comes back is the *published* forecast, not the boosters' raw output:
    `pooled.point_forecasts` caps each path's implied growth, smooths it, and
    scales each (sex, horizon) slice onto the share that sex held at the
    origin. The band calibration and the holdout scoring therefore measure the
    same forecasts the search page draws, which they would not if the stack
    were applied only to the production origin.
    """
    started = time.monotonic()
    training = pooled.training_rows(series, origin)
    models = pooled.train(training, threads=threads)
    rows = pooled.build_rows(series, [origin])
    predicted = pooled.point_forecasts(models, rows, pooled.growth_caps(training))
    note(
        f"  origin {origin}: {len(training):,} training rows, "
        f"{len(rows):,} names forecast ({time.monotonic() - started:.0f}s)"
    )
    return rows, predicted, len(training)


def _validation(row, predicted: np.ndarray, offsets: dict) -> dict | None:
    """Score one name's five-year holdout, and record what its bands covered.

    `skill` compares the model's holdout MAE against a naive/persistence
    baseline — the origin year's share repeated for every holdout year, the
    standard "no change" forecast. `skill = 1 - model_mae / naive_mae`: 0 means
    the model does no better than assuming nothing changes, negative means
    worse.

    `coverage` is this name's contribution to the population coverage figure;
    `_split_coverage` lifts it out before the payload is stored, so it never
    reaches the API.
    """
    if any(value is None for value in row["actual"]):
        return None

    actual = np.array(row["actual"], dtype=float)
    errors = actual - predicted
    mae = float(np.mean(np.abs(errors)))
    naive_mae = float(np.mean(np.abs(actual - row["last"])))

    coverage = {}
    for level, level_offsets in offsets.items():
        flags = []
        for horizon, value in enumerate(predicted):
            low, high = pooled.apply_band(float(value), level_offsets, horizon)
            flags.append(bool(low <= actual[horizon] <= high))
        coverage[level] = flags

    return {
        "mae": mae,
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mape": float(np.mean(np.abs(errors / np.maximum(actual, 1e-12))) * 100),
        "skill": float(1 - mae / naive_mae) if naive_mae > 0 else 0.0,
        "points": [
            {"year": int(row["origin"] + i + 1), "actual": float(a), "predicted": float(p)}
            for i, (a, p) in enumerate(zip(actual, predicted, strict=True))
        ],
        "coverage": coverage,
    }


def _split_coverage(stored: dict) -> tuple[dict[str, int], dict[str, int]]:
    """Take this name's coverage contribution out of the served payload.

    Coverage is a population statistic, not something the API should serve per
    name, so it moves to its own columns. See
    db_schema.CREATE_FORECASTS_TABLE.
    """
    hits: dict[str, int] = {}
    counts: dict[str, int] = {}
    validation = stored.get("validation")
    if validation is not None:
        for level, flags in validation.pop("coverage", {}).items():
            hits[level] = sum(flags)
            counts[level] = len(flags)
    return hits, counts


def _calibrate(conn) -> dict[str, float]:
    """Recompute `calibration` from every row in `forecasts`.

    Deliberately a function of the table's contents rather than of whatever
    this invocation happened to fit, so the published coverage always
    describes the whole stored population.
    """
    hits: dict[str, int] = defaultdict(int)
    counts: dict[str, int] = defaultdict(int)
    for stored_hits, stored_counts in conn.execute(
        "SELECT coverage_hits, coverage_n FROM forecasts "
        "WHERE coverage_hits IS NOT NULL AND coverage_n IS NOT NULL"
    ):
        for level, value in json.loads(stored_hits).items():
            hits[level] += value
        for level, value in json.loads(stored_counts).items():
            counts[level] += value

    conn.execute("DELETE FROM calibration")
    calibration = {}
    for level, n in counts.items():
        empirical = hits[level] / n if n else 0.0
        calibration[level] = empirical
        conn.execute(
            "INSERT INTO calibration (nominal_level, empirical_coverage, n) VALUES (?, ?, ?)",
            (float(level), empirical, n),
        )
    return calibration


def run(db_path: str, threads: int = pooled.THREADS, progress=None) -> dict:
    """Fit the pooled model and store a forecast for every eligible name/sex."""
    started = time.monotonic()

    def note(message: str) -> None:
        if progress is not None:
            progress(message)

    conn = sqlite3.connect(db_path)
    try:
        _ensure_schema(conn)
        (production_origin,) = conn.execute("SELECT MAX(year) FROM names").fetchone()
        holdout_origin = production_origin - pooled.H
        calibration_origin = holdout_origin - pooled.H

        # The whole extraction: one indexed pass, no sort file, no
        # intermediate artifact on disk. See pooled.SERIES_SQL.
        note("Streaming series from the names index...")
        series = list(pooled.stream_series(conn))
        note(f"  {len(series):,} name/sex series")

        note("Calibrating bands on the errors of an earlier origin...")
        cal_rows, cal_predicted, _ = _fit(series, calibration_origin, note, threads)
        offsets = pooled.band_offsets(pooled.log_residuals(cal_rows, cal_predicted), LEVELS)
        del cal_rows, cal_predicted

        note("Scoring the five-year holdout...")
        hold_rows, hold_predicted, _ = _fit(series, holdout_origin, note, threads)
        validations = {
            row["key"]: _validation(row, hold_predicted[i], offsets)
            for i, row in enumerate(hold_rows)
        }
        del hold_rows, hold_predicted

        note("Forecasting the years still to come...")
        rows, predicted, training_count = _fit(series, production_origin, note, threads)

        conn.execute("DELETE FROM forecasts")
        for i, row in enumerate(rows):
            name, sex = row["key"].rsplit("|", 1)
            stored = {
                "forecast": [
                    {
                        "year": int(production_origin + horizon + 1),
                        "mean": float(value),
                        **_band_fields(float(value), offsets, horizon),
                    }
                    for horizon, value in enumerate(predicted[i])
                ],
                "validation": validations.get(row["key"]),
            }
            hits, counts = _split_coverage(stored)
            conn.execute(
                "INSERT OR REPLACE INTO forecasts "
                "(name, sex, payload, coverage_hits, coverage_n) VALUES (?, ?, ?, ?, ?)",
                (name, sex, json.dumps(stored), json.dumps(hits), json.dumps(counts)),
            )

        calibration = _calibrate(conn)

        card = pooled.model_card(
            trained_through=production_origin,
            training_rows_count=training_count,
            training_origins=len(pooled.training_origins(production_origin)),
        )
        conn.execute("DELETE FROM model_card")
        conn.execute("INSERT INTO model_card (id, payload) VALUES (1, ?)", (json.dumps(card),))
        conn.commit()
    finally:
        conn.close()

    return {
        "eligible": len(rows),
        "stored": len(rows),
        "origins": {
            "production": production_origin,
            "holdout": holdout_origin,
            "calibration": calibration_origin,
        },
        "validated": sum(1 for v in validations.values() if v is not None),
        "seconds": time.monotonic() - started,
        "calibration": calibration,
        "model": card,
    }


def _band_fields(value: float, offsets: dict, horizon: int) -> dict[str, float]:
    """The two published bands around one forecast point.

    Both edges are the point times a positive multiplier, so neither can reach
    below zero — which is why the pipeline no longer clamps anything at zero.
    """
    fields = {}
    for level, key in ((0.8, "80"), (0.95, "95")):
        low, high = pooled.apply_band(value, offsets[str(level)], horizon)
        fields[f"lo{key}"] = low
        fields[f"hi{key}"] = high
    return fields


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db_path", nargs="?", default=DEFAULT_DB)
    parser.add_argument(
        "--threads",
        type=int,
        default=pooled.THREADS,
        help=(
            "Threads to train with (default: all cores). Pinned rather than dynamic "
            "because LightGBM is only bit-for-bit reproducible at a fixed thread count."
        ),
    )
    args = parser.parse_args()

    result = run(args.db_path, threads=args.threads, progress=lambda m: print(m, flush=True))

    origins = result["origins"]
    print(f"Model:              {result['model']['model_name']}")
    print(f"Trained on:         {result['model']['training_rows']:,} name-years")
    print(
        f"Origins:            production {origins['production']}, "
        f"holdout {origins['holdout']}, calibration {origins['calibration']}"
    )
    print(f"Forecasts stored:   {result['stored']:,}")
    print(f"With validation:    {result['validated']:,}")
    print(f"Took:               {result['seconds'] / 60:.1f}m")
    for level, coverage in sorted(result["calibration"].items()):
        print(f"Coverage @ {level:<5}     {coverage:.3f}")


if __name__ == "__main__":
    main()
