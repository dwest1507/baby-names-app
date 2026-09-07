"""Rolling-origin backtest.

Each base method is fitted once per name-origin; the shrunk and ensemble
variants are derived from those forecasts rather than refitted. Fits run in a
pebble pool under a per-name timeout for the same reason the app's precompute
does (`docs/adr/0007`): a handful of series send the ARIMA optimizer somewhere
it converges arbitrarily slowly, and unbounded they outlast the whole batch.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import TimeoutError as FutureTimeoutError  # noqa: E402

import numpy as np
from pebble import ProcessPool

SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SP)
import methods as M  # noqa: E402

from data import WORK, load  # noqa: E402

H, MIN_TRAIN, FLOOR = 5, 10, 1e-12

# Fitted on the whole observed history, as the app does today.
BASE = [
    "naive",
    "drift",
    "current",
    "arima_d1",
    "arima_log_d1",
    "ets_log_phi80",
    "loglin10_p70",
    "loglin7_p80",
    "loglin15_p90",
]

# Fitted on a trailing window of *calendar years* — not of observations, which
# would silently stretch the window across a gappy name's suppressed years.
WINDOWED = [
    ("current", 30),
    ("arima_log_d1", 30),
    ("arima_log_d1", 20),
    ("ets_log_phi80", 30),
    ("ets_log_phi80", 20),
]


def _log(x):
    return np.log(np.maximum(x, FLOOR))


def derive(b, last):
    """Shrunk and ensemble variants, from already-computed base forecasts."""
    out = {}
    ln = np.log(max(last, FLOOR))
    for m in ("current", "arima_d1", "arima_log_d1", "ets_log_phi80", "loglin10_p70"):
        for w in (0.5, 0.7):
            out[f"{m}_s{int(w * 100)}"] = np.exp(w * _log(b[m]) + (1 - w) * ln)
    out["ens_dampfam"] = np.exp(
        (_log(b["ets_log_phi80"]) + _log(b["loglin10_p70"]) + _log(b["loglin7_p80"])) / 3
    )
    out["ens_damp_arima"] = np.exp(
        (_log(b["ets_log_phi80"]) + _log(b["loglin10_p70"]) + _log(b["arima_d1"])) / 3
    )
    out["ens_all4"] = np.exp(
        (
            _log(b["ets_log_phi80"])
            + _log(b["loglin10_p70"])
            + _log(b["arima_d1"])
            + _log(b["current"])
        )
        / 4
    )
    out["ens_fixed3"] = np.exp(
        (_log(b["arima_log_d1_w30"]) + _log(b["ets_log_phi80"]) + _log(b["loglin10_p70"])) / 3
    )
    out["ens_damp_arima_s70"] = np.exp(0.7 * _log(out["ens_damp_arima"]) + 0.3 * ln)
    out["ens_dampfam_s70"] = np.exp(0.7 * _log(out["ens_dampfam"]) + 0.3 * ln)
    return out


def _safe(name, y, h, last):
    try:
        p = np.asarray(M.METHODS[name](y.copy(), h), dtype=float)
    except Exception:
        p = np.full(h, last)
    if p.shape != (h,) or not np.all(np.isfinite(p)):
        p = np.full(h, last)
    return np.maximum(p, 0.0)


def one(args):
    key, years, vals, rank, origins = args
    out = []
    for origin in origins:
        tr, te = years <= origin, (years > origin) & (years <= origin + H)
        ytr, yte = vals[tr], vals[te]
        if len(ytr) < MIN_TRAIN or len(yte) != H or years[tr][-1] != origin:
            continue
        last = float(ytr[-1])
        base, secs = {}, {}
        for m in BASE:
            t0 = time.perf_counter()
            base[m] = _safe(m, ytr, H, last)
            secs[m] = time.perf_counter() - t0
        for m, w in WINDOWED:
            win = vals[tr & (years > origin - w)]
            name = f"{m}_w{w}"
            t0 = time.perf_counter()
            base[name] = _safe(m, win, H, last) if len(win) >= MIN_TRAIN else base[m]
            secs[name] = time.perf_counter() - t0
        allm = dict(base)
        allm.update(derive(base, last))
        for m, p in allm.items():
            out.append(
                {
                    "key": str(key),
                    "rank": int(rank),
                    "origin": int(origin),
                    "method": m,
                    "secs": secs.get(m, 0.0),
                    "pred": np.maximum(p, 0.0).tolist(),
                    "actual": yte.tolist(),
                    "last": last,
                    "years": years[te].tolist(),
                }
            )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=None)
    ap.add_argument("--mid", type=int, default=0)
    ap.add_argument("--rest", type=int, default=0)
    ap.add_argument("--origins", default="2009,2014,2019")
    ap.add_argument("--workers", type=int, default=13)
    ap.add_argument("--timeout", type=float, default=90.0, help="seconds per name, all origins")
    ap.add_argument("--out", default=os.path.join(WORK, "main.jsonl"))
    a = ap.parse_args()

    origins = [int(x) for x in a.origins.split(",")]
    series = load(a.top, a.mid, a.rest)
    tasks = [(k, y, v, r, origins) for k, y, v, r in series]
    print(f"{len(tasks)} series x {len(origins)} origins", flush=True)

    t0 = time.time()
    abandoned = 0
    with open(a.out, "w") as fh, ProcessPool(max_workers=a.workers) as pool:
        future = pool.map(one, tasks, timeout=a.timeout, chunksize=4)
        results = future.result()
        for i in range(1, len(tasks) + 1):
            try:
                for r in next(results):
                    fh.write(json.dumps(r) + "\n")
            except StopIteration:
                break
            except FutureTimeoutError:
                abandoned += 1
            except Exception:
                abandoned += 1
            if i % 250 == 0:
                el = time.time() - t0
                print(
                    f"  {i}/{len(tasks)}  {el / 60:.1f}m  "
                    f"~{(len(tasks) - i) * el / i / 60:.1f}m left  ({abandoned} abandoned)",
                    flush=True,
                )
                fh.flush()
    print(f"done in {(time.time() - t0) / 60:.1f}m, {abandoned} abandoned -> {a.out}")


if __name__ == "__main__":
    main()
