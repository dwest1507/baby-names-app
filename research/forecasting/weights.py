"""Learned ensemble weights, per popularity tier and horizon.

The equal-weight log-space average in `combine.py` is a guess. This fits the
weights instead: on earlier origins, find the non-negative weights summing to 1
that best explain the log of what actually happened, per tier and per horizon,
then apply them to the held-out origin. Because `naive` is in the pool, the
shrinkage-toward-no-change that every good method needed is learned rather than
hard-coded.
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.optimize import nnls

SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SP)
from data import WORK, bucket  # noqa: E402

FLOOR = 1e-12
H = 5


def load_rows(paths):
    by = {}
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            by.setdefault((r["key"], r["origin"]), {})[r["method"]] = r
    return by


def design(by, origins, pool, tier, h):
    """Log-forecasts (X) and log-actuals (y) for one tier and horizon."""
    X, y = [], []
    for (_k, o), d in by.items():
        if o not in origins or not all(m in d for m in pool):
            continue
        if bucket(d[pool[0]]["rank"]) != tier:
            continue
        a = d[pool[0]]["actual"][h]
        if a is None or a <= 0:
            continue
        X.append([np.log(max(d[m]["pred"][h], FLOOR)) for m in pool])
        y.append(np.log(a))
    return np.array(X), np.array(y)


def simplex_nnls(X, y, ridge=1.0):
    """Non-negative weights summing to one (the sum is enforced by a heavy row)."""
    big = 1e3
    A = np.vstack([X, np.full((1, X.shape[1]), big), ridge * np.eye(X.shape[1])])
    b = np.concatenate([y, [big], np.zeros(X.shape[1])])
    w, _ = nnls(A, b)
    return w / w.sum() if w.sum() > 0 else np.full(X.shape[1], 1 / X.shape[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument(
        "--pool", default="naive,current,arima_log_d1_w30,ets_log_phi80,loglin10_p70,pooled2_full"
    )
    ap.add_argument("--fit-origins", default="2009,2014")
    ap.add_argument("--test-origin", type=int, default=2019)
    ap.add_argument("--name", default="ens_learned")
    ap.add_argument("--out", default=os.path.join(WORK, "learned.jsonl"))
    a = ap.parse_args()

    pool = a.pool.split(",")
    fit_origins = {int(x) for x in a.fit_origins.split(",")}
    by = load_rows(a.paths)
    tiers = ("top100", "top1000", "top5000", "rest")

    W = {}
    print(f"pool = {pool}\nweights fitted on origins {sorted(fit_origins)}:\n")
    for tier in tiers:
        for h in range(H):
            X, y = design(by, fit_origins, pool, tier, h)
            if len(X) < 50:
                continue
            W[(tier, h)] = simplex_nnls(X, y)
        if (tier, 4) in W:
            for label, h in (("h1", 0), ("h5", 4)):
                shown = " ".join(f"{m}={w:.2f}" for m, w in zip(pool, W[(tier, h)], strict=True))
                print(f"  {tier if h == 0 else '':8} {label} {shown}")

    rows, n = [], 0
    for (k, o), d in by.items():
        if o != a.test_origin or not all(m in d for m in pool):
            continue
        tier = bucket(d[pool[0]]["rank"])
        if (tier, 0) not in W:
            continue
        pred = []
        for h in range(H):
            lp = np.array([np.log(max(d[m]["pred"][h], FLOOR)) for m in pool])
            pred.append(float(np.exp(lp @ W[(tier, h)])))
        base = d[pool[0]]
        rows.append(
            {
                "key": k,
                "rank": base["rank"],
                "origin": o,
                "method": a.name,
                "secs": 0.0,
                "pred": pred,
                "actual": base["actual"],
                "last": base["last"],
                "years": base["years"],
            }
        )
        n += 1
    with open(a.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"\nwrote {n} rows -> {a.out}")


if __name__ == "__main__":
    main()
