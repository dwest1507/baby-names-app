"""A pooled ('global') forecaster: one model learned across all names.

Per-series ARIMA sees one name's 40-140 points. A pooled model sees the same
lifecycle shape repeated across 21,792 names and can learn regularities no
single-series model can (e.g. "a name near its all-time peak tends to fall").
Features and targets are in log space; a ridge is fit per horizon on
name-origins whose targets were already observable at the evaluation origin.
"""

import argparse
import json
import os
import sys

import numpy as np

SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SP)
from data import WORK, load, load_all  # noqa: E402

H = 5
FLOOR = 1e-9
FEATS = [
    "g1",
    "g2",
    "g3",
    "g5",
    "g10",
    "accel",
    "vol",
    "level",
    "below_peak",
    "yrs_since_peak",
    "age",
    "yrs_since_trough",
]


def features(ltrain):
    """ltrain: log values, oldest first, ending at the origin year."""
    n = len(ltrain)

    def g(k):
        k = min(k, n - 1)
        return (ltrain[-1] - ltrain[-1 - k]) / k

    d = np.diff(ltrain[-11:]) if n >= 3 else np.array([0.0])
    peak = int(np.argmax(ltrain))
    trough = int(np.argmin(ltrain[-30:])) if n >= 5 else 0
    return np.array(
        [
            g(1),
            g(2),
            g(3),
            g(5),
            g(10),
            g(1) - g(5),
            float(np.std(d)) if len(d) > 1 else 0.0,
            ltrain[-1],
            ltrain[-1] - ltrain[peak],
            min(n - 1 - peak, 60) / 60.0,
            min(n, 145) / 145.0,
            min(len(ltrain[-30:]) - 1 - trough, 30) / 30.0,
        ]
    )


def rows_for(series, origins, min_train=10):
    out = []
    for key, years, vals, rank in series:
        for o in origins:
            tr = years <= o
            if tr.sum() < min_train or years[tr][-1] != o:
                continue
            ltr = np.log(np.maximum(vals[tr], FLOOR))
            fut = {int(y): v for y, v in zip(years, vals, strict=True) if o < y <= o + H}
            tgt = np.full(H, np.nan)
            for i in range(1, H + 1):
                if (o + i) in fut:
                    tgt[i - 1] = np.log(max(fut[o + i], FLOOR)) - ltr[-1]
            out.append(
                {
                    "key": str(key),
                    "rank": int(rank),
                    "origin": int(o),
                    "x": features(ltr),
                    "y": tgt,
                    "last": float(vals[tr][-1]),
                    "actual": [fut.get(o + i) for i in range(1, H + 1)],
                    "years": [o + i for i in range(1, H + 1)],
                }
            )
    return out


def fit_ridge(X, y, lam):
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Z = np.hstack([np.ones((len(X), 1)), (X - mu) / sd])
    A = Z.T @ Z + lam * np.eye(Z.shape[1])
    A[0, 0] -= lam
    w = np.linalg.solve(A, Z.T @ y)
    return (w, mu, sd)


def predict_ridge(model, X):
    w, mu, sd = model
    Z = np.hstack([np.ones((len(X), 1)), (X - mu) / sd])
    return Z @ w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-origins", default="2014,2019")
    ap.add_argument("--lam", type=float, default=30.0)
    ap.add_argument("--name", default="pooled_ridge")
    ap.add_argument("--top", type=int, default=None)
    ap.add_argument("--mid", type=int, default=0)
    ap.add_argument("--rest", type=int, default=0)
    ap.add_argument("--train-pool", default="all", choices=["all", "top1000"])
    ap.add_argument("--out", default=os.path.join(WORK, "pooled.jsonl"))
    a = ap.parse_args()

    eval_origins = [int(x) for x in a.eval_origins.split(",")]

    # Training universe: every eligible series (cheap — no per-series fitting).
    all_series = load_all()
    if a.train_pool == "top1000":
        all_series = [s for s in all_series if s[3] <= 1000]
    eval_series = load(a.top, a.mid, a.rest)

    with open(a.out, "w") as fh:
        for eo in eval_origins:
            # Only origins whose 5-year targets were fully observable by `eo`.
            train_origins = list(range(1930, eo - H + 1))
            tr = rows_for(all_series, train_origins)
            X = np.vstack([r["x"] for r in tr])
            Y = np.vstack([r["y"] for r in tr])
            models = []
            for i in range(H):
                ok = ~np.isnan(Y[:, i])
                models.append(fit_ridge(X[ok], Y[ok, i], a.lam))
            ev = rows_for(eval_series, [eo])
            if not ev:
                continue
            Xe = np.vstack([r["x"] for r in ev])
            P = np.column_stack([predict_ridge(models[i], Xe) for i in range(H)])
            n_written = 0
            for r, p in zip(ev, P, strict=True):
                if any(v is None for v in r["actual"]):
                    continue
                pred = (np.exp(p) * r["last"]).tolist()
                fh.write(
                    json.dumps(
                        {
                            "key": r["key"],
                            "rank": r["rank"],
                            "origin": r["origin"],
                            "method": a.name,
                            "secs": 0.0,
                            "pred": pred,
                            "actual": r["actual"],
                            "last": r["last"],
                            "years": r["years"],
                        }
                    )
                    + "\n"
                )
                n_written += 1
            print(f"origin {eo}: trained on {len(X):,} name-origins, wrote {n_written}", flush=True)
            for i, (w, _, _) in enumerate(models):
                top = sorted(zip(FEATS, w[1:], strict=True), key=lambda t: -abs(t[1]))[:5]
                print(f"   h{i + 1} coef: " + ", ".join(f"{k}={v:+.3f}" for k, v in top))


if __name__ == "__main__":
    main()
