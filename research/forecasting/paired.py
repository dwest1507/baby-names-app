"""Is the gap between two methods bigger than the noise in the sample?

`poolSkill` is a ratio of two sums over a few hundred name-origins, and the top
tier is small enough that a handful of names move it. This resamples the
name-origins both methods cover, with replacement, and reports the distribution
of the difference — so a 0.04 gap on 198 names can be read for what it is.

    paired.py .work/cmp.jsonl --a gbt_pop_cap --b pooled2_pop_cap --bucket top100
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
    ap.add_argument("path")
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--buckets", default="top100,top1000,top5000,rest")
    ap.add_argument("--origin", type=int, default=None)
    ap.add_argument("--draws", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    by = defaultdict(dict)
    for line in open(a.path):
        r = json.loads(line)
        if r["method"] in (a.a, a.b, "naive"):
            by[(r["key"], r["origin"])][r["method"]] = r

    rng = np.random.default_rng(a.seed)
    print(
        f"{'bucket':10} {'n':>5} {'skill ' + a.a:>22} "
        f"{'skill ' + a.b:>22} {'diff [2.5%, 97.5%]':>28}"
    )
    for b in a.buckets.split(","):
        ea, eb, en = [], [], []
        for (_, origin), d in by.items():
            if len(d) < 3 or (a.origin is not None and origin != a.origin):
                continue
            if bucket(d["naive"]["rank"]) != b:
                continue
            act = np.array(d["naive"]["actual"], dtype=float)
            ea.append(np.abs(np.array(d[a.a]["pred"]) - act).sum())
            eb.append(np.abs(np.array(d[a.b]["pred"]) - act).sum())
            en.append(np.abs(d["naive"]["last"] - act).sum())
        if not ea:
            continue
        ea, eb, en = np.array(ea), np.array(eb), np.array(en)
        idx = rng.integers(0, len(ea), size=(a.draws, len(ea)))
        sa = 1 - ea[idx].sum(1) / en[idx].sum(1)
        sb = 1 - eb[idx].sum(1) / en[idx].sum(1)
        diff = sa - sb
        lo, hi = np.percentile(diff, [2.5, 97.5])
        alo, ahi = np.percentile(sa, [2.5, 97.5])
        blo, bhi = np.percentile(sb, [2.5, 97.5])
        print(
            f"{b:10} {len(ea):5d} "
            f"{1 - ea.sum() / en.sum():+9.3f} [{alo:+.3f},{ahi:+.3f}] "
            f"{1 - eb.sum() / en.sum():+9.3f} [{blo:+.3f},{bhi:+.3f}] "
            f"{diff.mean():+9.3f} [{lo:+.3f}, {hi:+.3f}]  P(a>b)={100 * (diff > 0).mean():.0f}%"
        )


if __name__ == "__main__":
    main()
