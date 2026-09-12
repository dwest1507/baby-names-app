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

The batch is one pass along the rolling backtest span — every origin from
1995 whose five-year window has since closed, 26 of them on the 2025 database
— followed by the production origin. Each origin gets its own fit, trained
only on windows that had already closed by then, so nothing it is scored on
was available to it.

Four things come out of that pass, and three of the origins in it have a
second job:

* every origin contributes each eligible name's five-year skill against the
  naive baseline, and those are averaged into the figure the search page
  labels the forecast line with. One window would mostly measure the
  2020-21 birth-rate shock; 26 measure the name. The same windows, summed per
  popularity tier, are what `model_evaluation` publishes for the deploy gate.
* `calibration` — ten years back. The spread of *its* errors sets the
  published bands, without those bands having seen the holdout they are then
  measured on. They are built per `(popularity tier, volatility bin)` and
  their coverage is measured over the same cells, so the figure in the
  `calibration` table is out-of-sample *and* describes names like the one
  being looked at rather than the average name. See
  docs/adr/0011-conformal-bands-keyed-by-strata.md.
* `holdout` — five years back, and the last origin of the span. Its
  predicted-against-actual points are the validation table the search page
  shows.
* `production` — the newest observed year. Its forecast is what gets served.

The fits share one `pooled.TrainingWindow`, which builds each origin's
features once and releases them as the span moves past — so 27 fits cost
little more feature extraction than one, and backtesting 26 origins holds no
more memory than fitting at a single one.

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


def _columns(conn, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _ensure_schema(conn) -> None:
    """Create the batch's tables, widening older ones where they are narrower.

    `CREATE TABLE IF NOT EXISTS` is a no-op against a table built before a
    column existed, so an artifact from an earlier run would keep its old
    shape and fail the insert. `forecasts` is widened in place because its
    rows are keyed by name and the batch replaces them; `calibration` is
    dropped and rebuilt instead, because its *key* changed — every run
    recomputes the whole table from `forecasts` anyway, so there is nothing in
    it to preserve.
    """
    conn.execute(db_schema.CREATE_FORECASTS_TABLE)
    conn.execute(db_schema.CREATE_MODEL_CARD_TABLE)
    conn.execute(db_schema.CREATE_MODEL_EVALUATION_TABLE)
    for column, kind in (
        ("coverage_hits", "TEXT"),
        ("coverage_n", "TEXT"),
        ("tier", "TEXT"),
        ("volatility_bin", "INTEGER"),
    ):
        if column not in _columns(conn, "forecasts"):
            conn.execute(f"ALTER TABLE forecasts ADD COLUMN {column} {kind}")

    existing = _columns(conn, "calibration")
    if existing and existing != set(db_schema.CALIBRATION_COLUMNS):
        conn.execute("DROP TABLE calibration")
    conn.execute(db_schema.CREATE_CALIBRATION_TABLE)


def _fit(window, origin: int, note, threads: int):
    """Train at one origin and forecast every name eligible there.

    The training rows come from the shared `pooled.TrainingWindow` rather than
    being built here, because 27 fits over a 40-origin window would otherwise
    extract the same features a thousand times over — feature extraction, not
    boosting, is what this batch spends its time on.

    What comes back is the *published* forecast, not the boosters' raw output:
    `pooled.point_forecasts` caps each path's implied growth, smooths it, and
    scales each (sex, horizon) slice onto the share that sex held at the
    origin. The band calibration, the skill measured at every backtest origin
    and the served forecast are therefore all the same object, which they
    would not be if the stack were applied only to the production origin.
    """
    started = time.monotonic()
    training, rows = window.advance(origin)
    models = pooled.train(training, threads=threads)
    predicted = pooled.point_forecasts(models, rows, pooled.growth_caps(training))
    note(
        f"  origin {origin}: {len(training):,} training rows, "
        f"{len(rows):,} names forecast ({time.monotonic() - started:.0f}s)"
    )
    return rows, predicted, len(training)


def _coverage_key(level, stratum) -> str:
    """`"0.8|top100|1"` — the cell a name's holdout points are counted into.

    Flattened into a string because it is stored as a JSON object key on the
    name's row; `_calibration_cell` reads it back.
    """
    tier, bin_index = stratum
    return f"{level}|{tier}|{bin_index}"


def _calibration_cell(key: str) -> tuple[float, str, int]:
    level, tier, bin_index = key.split("|")
    return float(level), tier, int(bin_index)


def _validation(row, predicted: np.ndarray, bands: dict, stratum) -> dict | None:
    """Score one name's five-year holdout, and record what its bands covered.

    The error figures are this one window's: the five years from the holdout
    origin, which are what the search page tabulates predicted against actual.
    `skill` is deliberately *not* among them — a single window's skill is
    mostly a measurement of that window, and the 2021-25 one contains the
    birth-rate shock. It is merged in afterwards from `pooled.BacktestTally`,
    averaged over every origin the name was eligible at since 1995.

    `coverage` is this name's contribution to the coverage figure for the cell
    whose band it was given — and `stratum` is the one it was in *at the
    holdout origin*, not the one it is served under, because what is being
    measured is how the bands performed for the names that were in that cell
    then. `_split_coverage` lifts it out before the payload is stored, so it
    never reaches the API.
    """
    if any(value is None for value in row["actual"]):
        return None

    actual = np.array(row["actual"], dtype=float)
    errors = actual - predicted

    coverage = {}
    for level in bands:
        offsets = pooled.band_for(bands, level, stratum)
        flags = []
        for horizon, value in enumerate(predicted):
            low, high = pooled.apply_band(float(value), offsets, horizon)
            flags.append(bool(low <= actual[horizon] <= high))
        # Counted into the cell whose band this name was handed, not into its
        # own — a stratum too thin to earn a band must not publish a coverage
        # figure for one. See `pooled.band_stratum`.
        coverage[_coverage_key(level, pooled.band_stratum(bands, level, stratum))] = flags

    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mape": float(np.mean(np.abs(errors / np.maximum(actual, 1e-12))) * 100),
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

    One row per `(level, stratum)` that the holdout actually populated, where
    a stratum is the cell whose *band* those names were given — so a row
    always reports the coverage of a band the names behind it held, and a
    stratum too thin to earn a band contributes to the `GLOBAL_STRATUM` row
    instead of publishing an estimate of its own. Returns the pooled coverage
    per level, which is the headline figure the CLI prints and is not stored:
    it describes no single band, so no name should ever be shown it.
    """
    hits: dict[str, int] = defaultdict(int)
    counts: dict[str, int] = defaultdict(int)
    for stored_hits, stored_counts in conn.execute(
        "SELECT coverage_hits, coverage_n FROM forecasts "
        "WHERE coverage_hits IS NOT NULL AND coverage_n IS NOT NULL"
    ):
        for key, value in json.loads(stored_hits).items():
            hits[key] += value
        for key, value in json.loads(stored_counts).items():
            counts[key] += value

    pooled_hits: dict[float, int] = defaultdict(int)
    pooled_counts: dict[float, int] = defaultdict(int)
    conn.execute("DELETE FROM calibration")
    for key, n in sorted(counts.items()):
        level, tier, bin_index = _calibration_cell(key)
        pooled_hits[level] += hits[key]
        pooled_counts[level] += n
        conn.execute(
            "INSERT INTO calibration "
            "(nominal_level, tier, volatility_bin, empirical_coverage, n) VALUES (?, ?, ?, ?, ?)",
            (level, tier, bin_index, hits[key] / n if n else 0.0, n),
        )

    return {
        str(level): pooled_hits[level] / n if n else 0.0
        for level, n in sorted(pooled_counts.items())
    }


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

        # One pass along the span, each origin fitted only on windows that had
        # already closed by then, then the production origin. `edges` and
        # `bands` are set at the calibration origin and `validations` at the
        # holdout origin, both of which come earlier in the sequence than the
        # points that read them.
        span = list(pooled.backtest_span(production_origin))
        scored = set(span)
        window = pooled.TrainingWindow(series)
        tally = pooled.BacktestTally()
        edges: list[float] = []
        bands: dict = {}
        validations: dict = {}

        # A database too short to reach 1995 has no span to score; the
        # calibration and holdout origins are still fitted, and the artifact
        # simply carries no measured skill rather than a fabricated one.
        note(
            f"Backtesting {len(span)} rolling origins"
            + (f" ({span[0]}:{span[-1]})..." if span else "...")
        )
        for origin in sorted(scored | {calibration_origin, holdout_origin}):
            rows, predicted, _ = _fit(window, origin, note, threads)
            if origin in scored:
                tally.add(origin, rows, predicted)
            if origin == calibration_origin:
                # The volatility bin edges are tertiles of the wobble present
                # at this origin, and they are fixed here and reused at every
                # later one, so that "bin 2" means the same thing whether a
                # name is being calibrated, scored, or served.
                edges = pooled.volatility_edges(rows)
                bands = pooled.strata_bands(rows, predicted, LEVELS, edges)
                note(
                    f"  {len(bands[str(LEVELS[0])]) - 1} of "
                    f"{len(pooled.TIERS) * pooled.VOLATILITY_BINS} "
                    "strata measured on their own errors"
                )
            if origin == holdout_origin:
                validations = {
                    row["key"]: _validation(
                        row, predicted[i], bands, pooled.row_stratum(row, edges)
                    )
                    for i, row in enumerate(rows)
                }
            del rows, predicted

        skills = tally.skill_per_name()
        evaluation = tally.evaluation()
        note(f"  {len(skills):,} names scored over {len(span)} origins")

        note("Forecasting the years still to come...")
        rows, predicted, training_count = _fit(window, production_origin, note, threads)

        conn.execute("DELETE FROM forecasts")
        for i, row in enumerate(rows):
            name, sex = row["key"].rsplit("|", 1)
            # The stratum this name is served under: its rank and its wobble
            # in the newest observed year. It decides both how wide the band
            # it is given is, and which measured coverage the chart may label
            # that band with.
            tier, volatility_bin = pooled.row_stratum(row, edges)
            stored = {
                "forecast": [
                    {
                        "year": int(production_origin + horizon + 1),
                        "mean": float(value),
                        **_band_fields(float(value), bands, (tier, volatility_bin), horizon),
                    }
                    for horizon, value in enumerate(predicted[i])
                ],
                # The holdout window's own errors, plus the skill measured
                # across the whole span. A name with a scored holdout has at
                # least that window in the tally — the holdout origin is the
                # span's last — so the two always arrive together.
                "validation": _with_skill(validations.get(row["key"]), skills, row["key"]),
            }
            hits, counts = _split_coverage(stored)
            conn.execute(
                "INSERT OR REPLACE INTO forecasts "
                "(name, sex, payload, coverage_hits, coverage_n, tier, volatility_bin) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    name,
                    sex,
                    json.dumps(stored),
                    json.dumps(hits),
                    json.dumps(counts),
                    tier,
                    volatility_bin,
                ),
            )

        calibration = _calibrate(conn)

        conn.execute("DELETE FROM model_evaluation")
        conn.executemany(
            "INSERT INTO model_evaluation "
            f"({', '.join(db_schema.MODEL_EVALUATION_COLUMNS)}) VALUES (?, ?, ?, ?, ?, ?)",
            [
                (tier, *(scores[column] for column in db_schema.MODEL_EVALUATION_COLUMNS[1:]))
                for tier, scores in sorted(evaluation.items())
            ],
        )

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
        "backtest": {"origins": span, "evaluation": evaluation, "names": len(skills)},
        "seconds": time.monotonic() - started,
        "calibration": calibration,
        "model": card,
    }


def _with_skill(validation: dict | None, skills: dict, key: str) -> dict | None:
    """Attach the span-wide skill to the holdout figures for one name.

    Kept separate from `_validation` because the two are measured over
    different things: everything else in the blob describes the holdout
    window, while `skill` describes the name across every window it was
    eligible for. `skill_windows` travels with it because it is what qualifies
    it — 26 measurements and one do not deserve equal weight, and the search
    page says which it is showing.
    """
    if validation is None:
        return None
    return {**validation, **skills[key]}


def _band_fields(value: float, bands: dict, stratum, horizon: int) -> dict[str, float]:
    """The two published bands around one forecast point, for this name's stratum.

    Both edges are the point times a positive multiplier, so neither can reach
    below zero — which is why the pipeline no longer clamps anything at zero.
    """
    fields = {}
    for level, key in ((0.8, "80"), (0.95, "95")):
        low, high = pooled.apply_band(value, pooled.band_for(bands, level, stratum), horizon)
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

    backtest = result["backtest"]
    origins = backtest["origins"]
    print(
        f"\nBacktest:           {len(origins)} origins "
        f"({origins[0]}:{origins[-1]}), {backtest['names']:,} names scored"
    )
    print(f"{'tier':<12}{'poolSkill':>11}{'medSkill':>10}{'origins':>9}")
    for tier in pooled.TIERS:
        scores = backtest["evaluation"].get(tier)
        if scores is not None:
            print(
                f"{tier:<12}{scores['pool_skill']:>11.3f}{scores['med_skill']:>10.3f}"
                f"{scores['origins_evaluated']:>9}"
            )


if __name__ == "__main__":
    main()
