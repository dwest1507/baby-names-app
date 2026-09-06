"""Conformal (split) prediction intervals from backtest residuals.

Calibrate log-scale residual quantiles on earlier origins, apply them at the
held-out origin, and measure the coverage actually achieved. Compares a single
global quantile against ones conditioned on popularity tier and on each
series's own recent volatility.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SP)
from data import SERIES, bucket  # noqa: E402

FLOOR = 1e-12


def vol_bin(v, edges):
    return int(np.searchsorted(edges, v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--method", default="ens_damp_arima")
    ap.add_argument("--cal-origins", default="2009,2014")
    ap.add_argument("--test-origin", type=int, default=2019)
    ap.add_argument("--levels", default="0.8,0.95")
    a = ap.parse_args()

    d = np.load(SERIES, allow_pickle=True)
    keys, years, vals = d["keys"], d["years"], d["vals"]
    kidx = {k: i for i, k in enumerate(keys)}

    def volatility(key, origin):
        i = kidx[key]
        y, v = years[i], vals[i]
        m = y <= origin
        lv = np.log(np.maximum(v[m][-11:], FLOOR))
        return float(np.std(np.diff(lv))) if len(lv) > 2 else 0.2

    cal_origins = {int(x) for x in a.cal_origins.split(",")}
    levels = [float(x) for x in a.levels.split(",")]

    rows = [json.loads(line) for line in open(a.path)]
    rows = [r for r in rows if r["method"] == a.method]
    for r in rows:
        p, ac = np.array(r["pred"]), np.array(r["actual"])
        r["res"] = np.log(np.maximum(p, FLOOR)) - np.log(np.maximum(ac, FLOOR))
        r["bucket"] = bucket(r["rank"])
        r["vol"] = volatility(r["key"], r["origin"])

    cal = [r for r in rows if r["origin"] in cal_origins]
    test = [r for r in rows if r["origin"] == a.test_origin]
    vols = np.array([r["vol"] for r in cal])
    edges = np.quantile(vols, [1 / 3, 2 / 3])
    print(
        f"method={a.method}  calibrate on {sorted(cal_origins)} (n={len(cal)}), "
        f"test {a.test_origin} (n={len(test)})\n"
    )

    def quantiles(sel, level):
        """Two-sided log-residual quantiles per horizon for a subset."""
        R = np.vstack([r["res"] for r in sel])
        lo = np.quantile(R, (1 - level) / 2, axis=0)
        hi = np.quantile(R, 1 - (1 - level) / 2, axis=0)
        return lo, hi

    schemes = {
        "global": lambda r: "*",
        "by_tier": lambda r: r["bucket"],
        "by_vol": lambda r: vol_bin(r["vol"], edges),
        "tier_x_vol": lambda r: (r["bucket"], vol_bin(r["vol"], edges)),
    }

    print(
        f"{'scheme':12} {'level':>6} {'cover@test':>11} {'cover top1000':>14} {'medWidthRatio':>14}"
    )
    for sname, keyf in schemes.items():
        for level in levels:
            groups = defaultdict(list)
            for r in cal:
                groups[keyf(r)].append(r)
            q = {k: quantiles(v, level) for k, v in groups.items() if len(v) >= 60}
            qg = quantiles(cal, level)
            hits, n, hits_t, n_t, widths = 0, 0, 0, 0, []
            for r in test:
                lo, hi = q.get(keyf(r), qg)
                p = np.maximum(np.array(r["pred"]), FLOOR)
                ac = np.array(r["actual"])
                low, high = p * np.exp(-hi), p * np.exp(-lo)
                inside = (ac >= low) & (ac <= high)
                hits += inside.sum()
                n += len(inside)
                widths.append(np.median(high / np.maximum(low, FLOOR)))
                if r["rank"] <= 1000:
                    hits_t += inside.sum()
                    n_t += len(inside)
            print(
                f"{sname:12} {level:6.2f} {hits / n:11.3f} "
                f"{hits_t / max(n_t, 1):14.3f} {np.median(widths):14.2f}"
            )

    # per-horizon detail for the global scheme
    print("\nper-horizon coverage (global scheme):")
    for level in levels:
        lo, hi = quantiles(cal, level)
        cov = []
        for h in range(5):
            ins = [
                (r["actual"][h] >= r["pred"][h] * np.exp(-hi[h]))
                and (r["actual"][h] <= r["pred"][h] * np.exp(-lo[h]))
                for r in test
            ]
            cov.append(np.mean(ins))
        print(
            f"  level {level}: "
            + " ".join(f"h{i + 1}={c:.3f}" for i, c in enumerate(cov))
            + "   half-width(log) "
            + " ".join(f"{(hi[i] - lo[i]) / 2:.2f}" for i in range(5))
        )


if __name__ == "__main__":
    main()
