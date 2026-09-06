"""Skill by origin: does a method's advantage hold across test windows, or is it one draw?

Rounds 1-3 tuned on two origins and tested on one. This reads any number of
forecast files and prints poolSkill as a method x origin matrix per tier, so a
conclusion drawn at 2019 can be checked against every other year the data
supports.

The naive baseline is recomputed per row from `last`, so a file needs no
`naive` rows of its own.

    origins.py .work/gbt_many.jsonl .work/ridge_many.jsonl --common
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SP)
from data import bucket  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--buckets", default="top100,top1000,top5000,rest")
    ap.add_argument(
        "--common",
        action="store_true",
        help="restrict to name-origins every method covers, for a like-for-like table",
    )
    ap.add_argument("--tiers-pooled", default="top100,top1000", help="tiers for the summary line")
    a = ap.parse_args()

    rows = [json.loads(line) for p in a.paths for line in open(p)]
    methods = sorted({r["method"] for r in rows})
    origins = sorted({r["origin"] for r in rows})

    covered = None
    if a.common:
        seen = defaultdict(set)
        for r in rows:
            seen[(r["key"], r["origin"])].add(r["method"])
        covered = {k for k, ms in seen.items() if len(ms) == len(methods)}
        print(f"common subset: {len(covered)} of {len(seen)} name-origins carry all {len(methods)}")

    # (method, origin, tier) -> [sum |model error|, sum |naive error|]
    acc = defaultdict(lambda: np.zeros(2))
    for r in rows:
        if covered is not None and (r["key"], r["origin"]) not in covered:
            continue
        act = np.array(r["actual"], dtype=float)
        acc[(r["method"], r["origin"], bucket(r["rank"]))] += (
            np.abs(np.array(r["pred"]) - act).sum(),
            np.abs(r["last"] - act).sum(),
        )

    def skill(method, origin, tiers):
        v = sum((acc[(method, origin, t)] for t in tiers), np.zeros(2))
        return 1 - v[0] / v[1] if v[1] > 0 else float("nan")

    for tier in a.buckets.split(","):
        print(f"\n=== {tier} — poolSkill by origin ===")
        print(f"{'method':16} " + " ".join(f"{o:>7}" for o in origins) + f" {'ALL':>8} {'mean':>7}")
        for m in methods:
            per = [skill(m, o, [tier]) for o in origins]
            allo = sum((acc[(m, o, tier)] for o in origins), np.zeros(2))
            pooled = 1 - allo[0] / allo[1] if allo[1] > 0 else float("nan")
            print(
                f"{m:16} "
                + " ".join(f"{v:7.3f}" for v in per)
                + f" {pooled:8.3f} {np.nanmean(per):7.3f}"
            )

    tiers = a.tiers_pooled.split(",")
    print(f"\n=== {'+'.join(tiers)} combined — poolSkill by origin ===")
    print(f"{'method':16} " + " ".join(f"{o:>7}" for o in origins) + f" {'ALL':>8} {'mean':>7}")
    for m in methods:
        per = [skill(m, o, tiers) for o in origins]
        allo = sum((acc[(m, o, t)] for o in origins for t in tiers), np.zeros(2))
        pooled = 1 - allo[0] / allo[1] if allo[1] > 0 else float("nan")
        print(
            f"{m:16} "
            + " ".join(f"{v:7.3f}" for v in per)
            + f" {pooled:8.3f} {np.nanmean(per):7.3f}"
        )

    if len(methods) == 2:
        m1, m2 = methods
        wins = sum(skill(m1, o, tiers) > skill(m2, o, tiers) for o in origins)
        print(f"\n{m1} beats {m2} on {wins} of {len(origins)} origins ({'+'.join(tiers)})")


if __name__ == "__main__":
    main()
