"""Per-name method selection: pick each name's method on an earlier origin's
holdout, then score that choice at the later origin. Also scores an oracle
(best-in-hindsight) as an upper bound on what selection can buy."""

import argparse
import json
from collections import defaultdict

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--pick-origin", type=int, default=2014)
    ap.add_argument("--test-origin", type=int, default=2019)
    ap.add_argument(
        "--pool", default="naive,current,arima_d1,ets_log_phi80,loglin10_p70,ens_damp_arima"
    )
    a = ap.parse_args()

    rows = [json.loads(line) for line in open(a.path)]
    by = defaultdict(dict)
    for r in rows:
        p, ac = np.array(r["pred"]), np.array(r["actual"])
        r["ae"] = np.abs(p - ac)
        by[(r["key"], r["origin"])][r["method"]] = r

    pool = a.pool.split(",")
    keys = {
        k
        for (k, o) in by
        if o == a.test_origin
        and (k, a.pick_origin) in by
        and all(
            m in by[(k, a.test_origin)] and m in by[(k, a.pick_origin)]
            for m in a.pool.split(",") + ["naive"]
        )
    }
    print(f"{len(keys)} names with both origins; pool={pool}")

    def score(label, chooser, sel_keys):
        num = den = 0.0
        per, picks = [], defaultdict(int)
        for k in sel_keys:
            te = by[(k, a.test_origin)]
            m = chooser(k, te)
            picks[m] += 1
            num += te[m]["ae"].sum()
            den += te["naive"]["ae"].sum()
            pm, pn = te[m]["ae"].mean(), te["naive"]["ae"].mean()
            per.append(1 - pm / pn if pn > 0 else 0.0)
        top = sorted(picks.items(), key=lambda t: -t[1])[:5]
        print(
            f"{label:26} poolSkill={1 - num / den:6.3f}  medSkill={np.median(per):6.3f}  "
            f"%>naive={100 * np.mean(np.array(per) > 0.001):5.1f}%  picks={top}"
        )

    for tier, sel in (
        ("ALL", keys),
        ("top1000", {k for k in keys if by[(k, a.test_origin)]["naive"]["rank"] <= 1000}),
        ("top100", {k for k in keys if by[(k, a.test_origin)]["naive"]["rank"] <= 100}),
    ):
        print(f"\n--- {tier} (n={len(sel)}) ---")
        for m in pool:
            score(f"fixed:{m}", lambda k, te, m=m: m, sel)

        def pick_on_earlier(k, te):
            earlier = by[(k, a.pick_origin)]
            return min(pool, key=lambda name: earlier[name]["ae"].mean())

        def pick_in_hindsight(k, te):
            return min(pool, key=lambda name: te[name]["ae"].mean())

        score("selected@pick-origin", pick_on_earlier, sel)
        score("oracle (hindsight)", pick_in_hindsight, sel)


if __name__ == "__main__":
    main()
