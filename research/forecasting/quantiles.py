"""Fit the forecast's prediction quantiles directly, with pinball loss.

Issue #34's first recommendation. Every interval measured so far is a
*residual* band: fit a point forecast, then paste symmetric log-residual
quantiles (conditioned on tier x volatility) around it — `conformal.py`. That
construction cannot express asymmetry. A name one year past its peak has a fat
downside and almost no upside, and a symmetric band is wrong in both
directions at once.

This fits the quantiles themselves. One LightGBM booster per (alpha, horizon)
on the same features, target and popularity weights as the point model, so the
only thing that changes between the arms is the loss.

The output carries both, so `intervals.py` can score residual bands, direct
quantiles and the conformalised hybrid on identical rows. `pred` and every
entry of `q` are on the share scale, like everything else in the harness:

    quantiles.py --eval-origins 1995:2019 --out .work/qr_many.jsonl
"""

import argparse
import json
import os
import sys
import time

import numpy as np

SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SP)
import pooled2  # noqa: E402
import pooled3  # noqa: E402

from data import load, load_all  # noqa: E402

H = pooled2.H


def train_quantiles(series, origin, sets, hp, alphas, weight, power, clip, seed=0):
    """`(alpha, horizon) -> booster`, plus the L2 point model for comparison.

    Every arm shares one feature matrix and one weight vector; only the
    objective differs, which is the whole point of the comparison.
    """
    import lightgbm as lgb

    tr = pooled2.train_rows(series, origin, sets, None, None)
    X = np.vstack([r["x"] for r in tr])
    Y = np.vstack([r["y"] for r in tr])
    w = pooled2.pop_weights(tr, power, clip) if weight == "pop" else None

    point, quant = [], {}
    for i in range(H):
        ok = ~np.isnan(Y[:, i])
        Xi, yi = X[ok], Y[ok, i]
        wi = None if w is None else w[ok]
        m = lgb.LGBMRegressor(**pooled3._lgb_params(hp, "l2", seed))
        m.fit(Xi, yi, sample_weight=wi)
        point.append(m)
        for a in alphas:
            p = pooled3._lgb_params(hp, "quantile", seed)
            p["alpha"] = a
            q = lgb.LGBMRegressor(**p)
            q.fit(Xi, yi, sample_weight=wi)
            quant[(a, i)] = q
    return point, quant, len(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", default="0.025,0.1,0.5,0.9,0.975")
    ap.add_argument("--sets", default="", help="comma list of: inter")
    ap.add_argument("--weight", default="pop", choices=["none", "pop"])
    ap.add_argument("--power", type=float, default=0.5)
    ap.add_argument("--clip", type=float, default=50.0)
    ap.add_argument("--leaves", type=int, default=15)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--trees", type=int, default=300)
    ap.add_argument("--min-child", type=int, default=200)
    ap.add_argument("--eval-origins", default="2009,2014,2019")
    ap.add_argument("--name", default="qr")
    ap.add_argument("--top", type=int, default=100000)
    ap.add_argument("--mid", type=int, default=1200)
    ap.add_argument("--rest", type=int, default=1200)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    alphas = [float(x) for x in a.alphas.split(",")]
    sets = {s for s in a.sets.split(",") if s}
    hp = {"leaves": a.leaves, "lr": a.lr, "trees": a.trees, "min_child": a.min_child}
    all_series = load_all()
    eval_series = load(a.top, a.mid, a.rest)

    out = open(a.out, "w") if a.out else None
    n_written = 0
    for eo in pooled3.parse_origins(a.eval_origins):
        pooled2.evict_train_rows()
        t0 = time.time()
        rows = pooled2.rows_for(eval_series, [eo], sets, None)
        point, quant, n_train = train_quantiles(
            all_series, eo, sets, hp, alphas, a.weight, a.power, a.clip
        )
        X = np.vstack([r["x"] for r in rows])
        last = np.array([r["last"] for r in rows])
        P = np.column_stack([m.predict(X) for m in point])
        Q = {aa: np.column_stack([quant[(aa, i)].predict(X) for i in range(H)]) for aa in alphas}

        print(
            f"\n{a.name} origin {eo} ({n_train:,} training name-origins, "
            f"{time.time() - t0:.0f}s, {len(alphas) * H + H} boosters)"
        )
        pooled3.show(pooled2.evaluate(rows, list(np.exp(P) * last[:, None])), "point (l2)")
        if 0.5 in Q:
            pooled3.show(
                pooled2.evaluate(rows, list(np.exp(Q[0.5]) * last[:, None])), "median (q50)"
            )

        for j, r in enumerate(rows):
            if any(v is None for v in r["actual"]):
                continue
            out_row = {
                "key": r["key"],
                "rank": r["rank"],
                "origin": r["origin"],
                "method": a.name,
                "secs": 0.0,
                "pred": [float(v) for v in np.exp(P[j]) * r["last"]],
                "q": {str(aa): [float(v) for v in np.exp(Q[aa][j]) * r["last"]] for aa in alphas},
                "actual": r["actual"],
                "last": r["last"],
                "years": [r["origin"] + i for i in range(1, H + 1)],
            }
            if out:
                out.write(json.dumps(out_row) + "\n")
            n_written += 1
        if out:
            out.flush()

    if out:
        out.close()
        print(f"\nwrote {n_written} rows -> {a.out}")


if __name__ == "__main__":
    main()
