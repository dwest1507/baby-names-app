"""Cap a forecast's implied growth, and write the capped variants.

A log-scale model can extrapolate multiplicative growth without limit: the
true-log ARIMA arm produced a five-year ratio of 2.5e44 for 9 of ~13,000
name-origins. Rare, but one such forecast is a visibly broken chart, and it
drags any error sum it lands in.

The cap is read off the data rather than invented: the largest five-year change
any name actually made, at the origins used for fitting. Anything beyond that
is not a forecast, it is a divergence.
"""

import argparse
import json
import os
import sys

import numpy as np

SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SP)
from data import WORK  # noqa: E402

FLOOR = 1e-12
H = 5


def caps_from(rows, origins, quantile):
    """Per-horizon bound on |log(value / last observed)|, from observed history."""
    seen = set()
    obs = []
    for r in rows:
        if r["origin"] not in origins or (r["key"], r["origin"]) in seen:
            continue
        seen.add((r["key"], r["origin"]))
        last = max(r["last"], FLOOR)
        a = np.maximum(np.array(r["actual"], dtype=float), FLOOR)
        obs.append(np.abs(np.log(a / last)))
    return np.quantile(np.vstack(obs), quantile, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--fit-origins", default="2009,2014")
    ap.add_argument("--quantile", type=float, default=0.999)
    ap.add_argument("--methods", default="", help="comma list; default every method present")
    ap.add_argument("--out", default=os.path.join(WORK, "capped.jsonl"))
    a = ap.parse_args()

    rows = [json.loads(line) for p in a.paths for line in open(p)]
    caps = caps_from(rows, {int(x) for x in a.fit_origins.split(",")}, a.quantile)
    print("caps on |log(pred/last)|: " + " ".join(f"h{i + 1}={c:.2f}" for i, c in enumerate(caps)))

    wanted = set(a.methods.split(",")) if a.methods else {r["method"] for r in rows}
    n = clipped = 0
    with open(a.out, "w") as fh:
        for r in rows:
            if r["method"] not in wanted or r["method"] == "naive":
                continue
            last = max(r["last"], FLOOR)
            lc = np.log(np.maximum(np.array(r["pred"], dtype=float), FLOOR) / last)
            new = np.clip(lc, -caps, caps)
            clipped += int(np.any(new != lc))
            fh.write(
                json.dumps({**r, "method": r["method"] + "_cap", "pred": list(np.exp(new) * last)})
                + "\n"
            )
            n += 1
    print(f"wrote {n} rows ({clipped} had at least one horizon clipped) -> {a.out}")


if __name__ == "__main__":
    main()
