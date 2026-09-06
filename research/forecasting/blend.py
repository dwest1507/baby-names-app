"""Blend two forecasts in log space, with the weight chosen on earlier origins.

`combine.py` averages a fixed set of methods with equal weights. This picks the
weight instead of assuming it, on the same terms as every other tuning decision
here: the grid is scored on `--fit-origins` only, and the chosen weight is then
applied to every origin present so the held-out one stays untouched.

    blend.py .work/pooled2_pop.jsonl .work/gbt_life.jsonl \\
             --pair pooled2_pop,gbt_life --name blend_ridge_gbt \\
             --fit-origins 2009,2014 --out .work/blend.jsonl
"""

import argparse
import json
import os
import sys

import numpy as np

SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SP)
from data import WORK, bucket  # noqa: E402

FLOOR = 1e-12
KEEP = ("key", "rank", "origin", "actual", "last", "years")


def pool_skill(rows, tiers):
    """Skill against the naive baseline, pooled over the tiers that matter."""
    num = den = 0.0
    for pred, last, actual, rank in rows:
        if bucket(rank) not in tiers:
            continue
        a = np.array(actual, dtype=float)
        num += np.abs(np.array(pred) - a).sum()
        den += np.abs(last - a).sum()
    return 1 - num / den if den > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--pair", required=True, help="two method names, comma separated")
    ap.add_argument("--name", default=None)
    ap.add_argument("--fit-origins", default="2009,2014")
    ap.add_argument("--weights", default="0,0.25,0.4,0.5,0.6,0.75,1")
    ap.add_argument("--tiers", default="top100,top1000")
    ap.add_argument("--out", default=os.path.join(WORK, "blend.jsonl"))
    a = ap.parse_args()

    m1, m2 = a.pair.split(",")
    name = a.name or f"blend_{m1}_{m2}"
    tiers = set(a.tiers.split(","))
    fit = {int(x) for x in a.fit_origins.split(",")}

    by = {}
    for p in a.paths:
        for line in open(p):
            r = json.loads(line)
            if r["method"] in (m1, m2):
                by.setdefault((r["key"], r["origin"]), {})[r["method"]] = r
    both = [d for d in by.values() if m1 in d and m2 in d]
    print(f"{len(both)} name-origins carry both {m1} and {m2}")

    def blended(d, w):
        return np.exp(
            w * np.log(np.maximum(d[m1]["pred"], FLOOR))
            + (1 - w) * np.log(np.maximum(d[m2]["pred"], FLOOR))
        )

    best = (-np.inf, None)
    for w in [float(x) for x in a.weights.split(",")]:
        rows = [
            (blended(d, w), d[m1]["last"], d[m1]["actual"], d[m1]["rank"])
            for d in both
            if d[m1]["origin"] in fit
        ]
        if not rows:
            raise SystemExit(f"no rows at fit origins {sorted(fit)}")
        s = pool_skill(rows, tiers)
        print(f"  w({m1})={w:<5} poolSkill@fit={s:+.4f}{'  *' if s > best[0] else ''}")
        if s > best[0]:
            best = (s, w)
    w = best[1]
    print(f"  chosen w({m1})={w}")

    with open(a.out, "w") as fh:
        for d in both:
            base = d[m1]
            fh.write(
                json.dumps(
                    {
                        **{k: base[k] for k in KEEP},
                        "method": name,
                        "secs": 0.0,
                        "pred": blended(d, w).tolist(),
                    }
                )
                + "\n"
            )
    print(f"wrote {len(both)} rows -> {a.out}")


if __name__ == "__main__":
    main()
