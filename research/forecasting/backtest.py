"""Rolling-origin backtest, computing each base method once per name-origin and
deriving the shrunk/ensemble variants from those forecasts (no refitting)."""

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SP)
import methods as M  # noqa: E402

from data import WORK, load  # noqa: E402

H, MIN_TRAIN, FLOOR = 5, 10, 1e-12
BASE = [
    "naive",
    "drift",
    "current",
    "arima_d1",
    "ets_log_phi80",
    "loglin10_p70",
    "loglin7_p80",
    "loglin15_p90",
]


def _log(x):
    return np.log(np.maximum(x, FLOOR))


def derive(b, last):
    """Shrunk and ensemble variants, from already-computed base forecasts."""
    out = {}
    ln = np.log(max(last, FLOOR))
    for m in ("current", "arima_d1", "ets_log_phi80", "loglin10_p70"):
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
    out["ens_damp_arima_s70"] = np.exp(0.7 * _log(out["ens_damp_arima"]) + 0.3 * ln)
    out["ens_dampfam_s70"] = np.exp(0.7 * _log(out["ens_dampfam"]) + 0.3 * ln)
    return out


def one(args):
    key, years, vals, rank, origins = args
    out = []
    for origin in origins:
        tr, te = years <= origin, (years > origin) & (years <= origin + H)
        ytr, yte = vals[tr], vals[te]
        if len(ytr) < MIN_TRAIN or len(yte) != H or years[tr][-1] != origin:
            continue
        base, secs = {}, {}
        for m in BASE:
            t0 = time.perf_counter()
            try:
                p = np.asarray(M.METHODS[m](ytr.copy(), H), dtype=float)
            except Exception:
                p = np.full(H, ytr[-1])
            if p.shape != (H,) or not np.all(np.isfinite(p)):
                p = np.full(H, ytr[-1])
            base[m] = np.maximum(p, 0.0)
            secs[m] = time.perf_counter() - t0
        allm = dict(base)
        allm.update(derive(base, float(ytr[-1])))
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
                    "last": float(ytr[-1]),
                    "years": years[te].tolist(),
                }
            )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=None)
    ap.add_argument("--mid", type=int, default=0)
    ap.add_argument("--rest", type=int, default=0)
    ap.add_argument("--origins", default="2014,2019")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--out", default=os.path.join(WORK, "main.jsonl"))
    a = ap.parse_args()
    origins = [int(x) for x in a.origins.split(",")]
    series = load(a.top, a.mid, a.rest)
    tasks = [(k, y, v, r, origins) for k, y, v, r in series]
    print(f"{len(tasks)} series x {len(origins)} origins", flush=True)
    t0 = time.time()
    with open(a.out, "w") as fh, Pool(a.workers) as pool:
        for i, rows in enumerate(pool.imap_unordered(one, tasks, chunksize=4), 1):
            for r in rows:
                fh.write(json.dumps(r) + "\n")
            if i % 250 == 0:
                el = time.time() - t0
                print(
                    f"  {i}/{len(tasks)}  {el / 60:.1f}m  "
                    f"~{(len(tasks) - i) * el / i / 60:.1f}m left",
                    flush=True,
                )
                fh.flush()
    print(f"done in {(time.time() - t0) / 60:.1f}m -> {a.out}")


if __name__ == "__main__":
    main()
