"""Precompute forecasts for every eligible name/sex and store them.

Forecasting is CPU-bound (one ARIMA grid search, twice, per name) and the
source data only refreshes once a year, so fitting on every request buys
nothing. This script does it once, offline, and writes the result into the
`forecasts` table so the API becomes a lookup. See
docs/adr/0004-forecasts-as-a-build-artifact.md.

It reads the whole `names` table once and groups it in memory by (lowercased
name, sex) rather than querying per name — with ~24,700 eligible pairs in the
real database, one query per name would pay the lookup cost that many times
over. Fitting itself is delegated to `app.services.forecast.fit_forecast`,
the same code the API route used to call at request time, so stored values
cannot drift from what the code would produce live.

Runnable against either the sample database (fast, a handful of names) or the
real built database (~24,700 eligible pairs). On the real database this is a
batch/offline job measured in tens of minutes, not something run in a hot
loop. Three things keep it that way, and each has a cost worth knowing about
— see docs/adr/0007-precompute-batch-runs-in-parallel.md:

* `--workers` fans the fits across processes. Names are independent, so this
  is pure throughput with no effect on results.
* `--timeout` bounds a single name. A small number of series send the ARIMA
  optimizer somewhere it converges extremely slowly; unbounded, one of them
  can outlast the entire rest of the batch while every other worker idles.
* `--resume` skips name/sex pairs already stored, so a crash costs the names
  still outstanding rather than the whole run.

Usage: uv run python scripts/precompute_forecasts.py [db_path] [--workers N]
                                                     [--timeout SECONDS]
                                                     [--resume]
"""

import os

# Every worker fits tiny series — a few dozen points — where a threaded BLAS is
# pure overhead, and N worker processes each spawning N BLAS threads oversubscribes
# the machine badly enough to land slower than serial. This must happen before
# numpy is imported anywhere, hence before the app imports below.
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import sqlite3  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402
from concurrent.futures import TimeoutError as FutureTimeoutError  # noqa: E402
from pathlib import Path  # noqa: E402

from pebble import ProcessPool  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent))

from app import db_schema  # noqa: E402
from app.services import forecast  # noqa: E402

REPO_ROOT = Path(__file__).parent.parent.parent
DEFAULT_DB = str(REPO_ROOT / "data" / "names.built.db")
DEFAULT_TIMEOUT = 30.0
COMMIT_EVERY = 500


# A name that outruns `--timeout` is killed at the process level, by pebble,
# rather than interrupted inside the worker.
#
# The in-process approach does not work here and it is worth recording why: a
# `signal.setitimer` alarm is only delivered between bytecode instructions, and
# a runaway `ARIMA.fit()` stays inside compiled statsmodels/scipy code for
# minutes at a time without returning to the interpreter. The alarm is never
# handled, so the cap silently does nothing to the only names it exists for. A
# first attempt at this also derived the timeout from `Exception`, which
# `_fit_best_model`'s `except Exception: continue` swallowed outright. Measured
# on the real database, ~0.1% of names run away (2 in a 2,000-name sample), and
# roughly 25 of them are enough to occupy every worker indefinitely.
#
# Killing the worker is the only thing that reliably stops compiled code, so
# the batch takes the process cost and lets pebble restart the worker.


def _fit_one(history: list[dict]) -> dict | None:
    """Fit one name/sex. Runs in a pebble worker, under a hard timeout."""
    return forecast.fit_forecast(history)


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
        coverage = validation.pop("coverage", None)
        if coverage:
            for level, flags in coverage.items():
                hits[level] = sum(flags)
                counts[level] = len(flags)
    return hits, counts


def _ensure_schema(conn) -> None:
    """Create the batch's tables, and add the coverage columns to an older one.

    `CREATE TABLE IF NOT EXISTS` is a no-op against a `forecasts` table built
    before coverage was stored per name, so an artifact from an earlier run
    would keep its three columns and fail the insert. Adding the columns is
    enough of a migration: they are nullable, and `_calibrate` ignores rows
    where they are NULL.
    """
    conn.execute(db_schema.CREATE_FORECASTS_TABLE)
    conn.execute(db_schema.CREATE_CALIBRATION_TABLE)
    existing = {row[1] for row in conn.execute("PRAGMA table_info(forecasts)")}
    for column in ("coverage_hits", "coverage_n"):
        if column not in existing:
            conn.execute(f"ALTER TABLE forecasts ADD COLUMN {column} TEXT")


def _calibrate(conn) -> dict[str, float]:
    """Recompute `calibration` from every row in `forecasts`.

    Deliberately a function of the table's contents rather than of whatever
    this invocation happened to fit, so that a resumed or partial run still
    publishes a coverage figure describing the whole population. See
    docs/adr/0007-precompute-batch-runs-in-parallel.md.
    """
    hits: dict[str, int] = defaultdict(int)
    counts: dict[str, int] = defaultdict(int)
    for row in conn.execute(
        "SELECT coverage_hits, coverage_n FROM forecasts "
        "WHERE coverage_hits IS NOT NULL AND coverage_n IS NOT NULL"
    ):
        for level, value in json.loads(row[0]).items():
            hits[level] += value
        for level, value in json.loads(row[1]).items():
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


def run(
    db_path: str,
    workers: int = 1,
    timeout: float = DEFAULT_TIMEOUT,
    resume: bool = False,
    progress=None,
) -> dict:
    """Fit and store a forecast for every eligible name/sex pair in db_path.

    `workers` defaults to 1 — serial, no process pool — so that importers (the
    test suite builds its sample database through this function) get simple,
    in-process behaviour. Only the CLI fans out.
    """
    started = time.monotonic()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        _ensure_schema(conn)

        (latest_year,) = conn.execute("SELECT MAX(year) FROM names").fetchone()

        # The whole table, read once, then grouped in memory — not one query
        # per name/sex pair.
        rows = conn.execute(
            "SELECT name, sex, year, total_count, popularity_percent, popularity_rank "
            "FROM names ORDER BY name, sex, year"
        ).fetchall()

        groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in rows:
            groups[(row["name"].lower(), row["sex"])].append(dict(row))
        del rows

        already = set()
        if resume:
            already = {(r[0], r[1]) for r in conn.execute("SELECT name, sex FROM forecasts")}
        else:
            conn.execute("DELETE FROM forecasts")

        tasks = [
            (key, history)
            for key, history in groups.items()
            if key not in already
            and forecast.is_eligible([r["year"] for r in history], latest_year)
        ]
        total = len(tasks)
        # Nothing downstream needs the full grouping, and holding it while the
        # pool forks would hand every worker a copy of the entire names table.
        del groups

        stored_count = 0
        timed_out = 0
        abandoned: list[tuple[str, str]] = []

        def record(key: tuple[str, str], stored: dict | None) -> None:
            nonlocal stored_count, timed_out
            if stored is None:
                timed_out += 1
                abandoned.append(key)
                return
            hits, counts = _split_coverage(stored)
            conn.execute(
                "INSERT OR REPLACE INTO forecasts "
                "(name, sex, payload, coverage_hits, coverage_n) VALUES (?, ?, ?, ?, ?)",
                (key[0], key[1], json.dumps(stored), json.dumps(hits), json.dumps(counts)),
            )
            stored_count += 1

        def checkpoint(done: int) -> None:
            if done % COMMIT_EVERY == 0:
                conn.commit()
            if progress is not None and (done % COMMIT_EVERY == 0 or done == total):
                progress(done, total, timed_out, time.monotonic() - started)

        # One code path for every worker count, so `workers=1` — what the test
        # suite and any importer get — exercises the same timeout and the same
        # result handling as the CLI's fan-out. Results arrive in submission
        # order, which is fine now that no single name can outlast `timeout`.
        with ProcessPool(max_workers=max(1, workers)) as pool:
            future = pool.map(_fit_one, [history for _key, history in tasks], timeout=timeout)
            results = future.result()
            for done, (key, _history) in enumerate(tasks, start=1):
                try:
                    record(key, next(results))
                except StopIteration:
                    break
                except FutureTimeoutError:
                    # pebble has already killed and replaced the worker.
                    record(key, None)
                except Exception:
                    logging.getLogger(__name__).warning(
                        "forecast failed for %s/%s", *key, exc_info=True
                    )
                    record(key, None)
                checkpoint(done)

        calibration = _calibrate(conn)
        conn.commit()
    finally:
        conn.close()

    return {
        "eligible": total,
        "stored": stored_count,
        "timed_out": timed_out,
        "abandoned": abandoned,
        "resumed_skips": len(already),
        "seconds": time.monotonic() - started,
        "calibration": calibration,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db_path", nargs="?", default=DEFAULT_DB)
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Processes to fit with (default: all cores). 1 runs serially, in-process.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=(
            "Seconds to allow one name before abandoning it and storing no forecast "
            f"(default: {DEFAULT_TIMEOUT:g})."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep forecasts already stored and fit only the outstanding names.",
    )
    args = parser.parse_args()

    def progress(done: int, total: int, abandoned: int, elapsed: float) -> None:
        rate = done / elapsed if elapsed else 0.0
        remaining = (total - done) / rate if rate else 0.0
        print(
            f"  {done:,}/{total:,} names  {elapsed / 60:.1f}m elapsed  "
            f"~{remaining / 60:.1f}m remaining  ({abandoned} abandoned)",
            flush=True,
        )

    result = run(
        args.db_path,
        workers=args.workers,
        timeout=args.timeout,
        resume=args.resume,
        progress=progress,
    )
    print(f"Eligible this run:    {result['eligible']:,}")
    print(f"Forecasts stored:     {result['stored']:,}")
    print(f"Abandoned (timeout):  {result['timed_out']:,}")
    for name, sex in result["abandoned"][:20]:
        print(f"                        {name} ({sex})")
    if result["resumed_skips"]:
        print(f"Skipped (resume):     {result['resumed_skips']:,}")
    print(f"Took:                 {result['seconds'] / 60:.1f}m")
    for level, coverage in sorted(result["calibration"].items()):
        print(f"Coverage @ {level}:        {coverage:.3f}")


if __name__ == "__main__":
    main()
