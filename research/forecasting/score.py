"""Score a backtest results.jsonl."""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import bucket


def load(path):
    rows = [json.loads(line) for line in open(path)]
    for r in rows:
        p = np.array(r["pred"])
        a = np.array(r["actual"])
        r["ae"] = np.abs(p - a)
        r["ape"] = np.abs((p - a) / np.maximum(a, 1e-12))
        r["logerr"] = np.log(np.maximum(p, 1e-12)) - np.log(np.maximum(a, 1e-12))
        r["bucket"] = bucket(r["rank"])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--buckets", default="top100,top1000,top5000,rest,ALL")
    ap.add_argument("--by-horizon", action="store_true")
    ap.add_argument("--by-origin", action="store_true")
    a = ap.parse_args()
    rows = load(a.path)

    idx = defaultdict(dict)  # (key,origin) -> method -> row
    for r in rows:
        idx[(r["key"], r["origin"])][r["method"]] = r
    methods = sorted({r["method"] for r in rows})
    origins = sorted({r["origin"] for r in rows})

    def report(sel_rows, title):
        by = defaultdict(list)
        for d in idx.values():
            if "naive" not in d:
                continue
            base = d["naive"]
            if not sel_rows(base):
                continue
            for m, r in d.items():
                by[m].append((r, base))
        print(f"\n=== {title}  (n={len(by.get('naive', []))} name-origins) ===")
        print(
            f"{'method':20} {'poolSkill':>9} {'medSkill':>9} {'%>naive':>8} "
            f"{'medMAPE%':>9} {'medLogAE':>9} {'p90APE%':>8} {'secs':>7}"
        )
        base_pool = sum(r[1]["ae"].sum() for r in by["naive"])
        out = []
        for m in methods:
            rs = by[m]
            if not rs:
                continue
            pool = 1 - sum(r["ae"].sum() for r, _ in rs) / base_pool
            per = [
                1 - r["ae"].mean() / b["ae"].mean() if b["ae"].mean() > 0 else 0.0 for r, b in rs
            ]
            ape = np.concatenate([r["ape"] for r, _ in rs]) * 100
            lae = np.abs(np.concatenate([r["logerr"] for r, _ in rs]))
            out.append(
                (
                    pool,
                    m,
                    np.median(per),
                    100 * np.mean(np.array(per) > 0.001),
                    np.median(ape),
                    np.median(lae),
                    np.percentile(ape, 90),
                    np.mean([r["secs"] for r, _ in rs]),
                )
            )
        for pool, m, med, pct, mape, lae, p90, secs in sorted(out, reverse=True):
            print(
                f"{m:20} {pool:9.3f} {med:9.3f} {pct:7.1f}% "
                f"{mape:9.1f} {lae:9.4f} {p90:8.1f} {secs:7.2f}"
            )

    for b in a.buckets.split(","):
        if b == "ALL":
            report(lambda r: True, "ALL")
        else:
            report(lambda r, b=b: r["bucket"] == b, b)

    if a.by_origin:
        for o in origins:
            report(lambda r, o=o: r["origin"] == o and r["rank"] <= 1000, f"top1000 @ origin {o}")

    if a.by_horizon:
        print("\n=== MAE skill vs naive by horizon (rank<=1000) ===")
        print(f"{'method':20} " + " ".join(f"{'h' + str(i + 1):>7}" for i in range(5)))
        sel = [(k, o) for (k, o), d in idx.items() if d["naive"]["rank"] <= 1000]
        bh = np.array([idx[k]["naive"]["ae"] for k in sel]).sum(axis=0)
        for m in methods:
            mh = np.array([idx[k][m]["ae"] for k in sel if m in idx[k]]).sum(axis=0)
            print(f"{m:20} " + " ".join(f"{1 - mh[i] / bh[i]:7.3f}" for i in range(5)))


if __name__ == "__main__":
    main()
