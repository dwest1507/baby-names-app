"""Pooled ('global') forecaster, second generation.

Adds to `pooled.py`: cross-name cohort features, level interactions (so popular
and rare names can behave differently inside one fit), optional
popularity-weighted fitting, and a ridge penalty chosen on an earlier origin
rather than guessed. Scores itself against the naive baseline so an ablation can
run without touching the per-series backtest.

Nothing is fitted on data later than the origin being forecast: training rows
are name-origins whose five-year targets had already been observed by then.
"""

import argparse
import json
import os
import sys

import numpy as np

SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SP)
from cohorts import Cohorts  # noqa: E402

from data import bucket, load, load_all  # noqa: E402

H = 5
FLOOR = 1e-9

BASE_FEATS = [
    "g1",
    "g2",
    "g3",
    "g5",
    "g10",
    "accel",
    "vol",
    "level",
    "level2",
    "below_peak",
    "yrs_since_peak",
    "age",
    "yrs_since_trough",
]
INTER_FEATS = ["lvl_g1", "lvl_g3", "lvl_g5", "lvl_g10", "lvl_accel", "lvl_peak"]
COHORT_FEATS = ["ce_g3", "ce_g10", "ci_g5", "cs_g5", "rel_g3", "ce_level", "own_frac"]


def _growth(ly, k):
    k = min(k, len(ly) - 1)
    return (ly[-1] - ly[-1 - k]) / k if k >= 1 else 0.0


def features(ltrain, cohort_hist, sets):
    """`cohort_hist` is the leave-one-out cohort share history, oldest first."""
    n = len(ltrain)
    d = np.diff(ltrain[-11:]) if n >= 3 else np.array([0.0])
    peak = int(np.argmax(ltrain))
    trough = int(np.argmin(ltrain[-30:])) if n >= 5 else 0
    g1, g2, g3, g5, g10 = (_growth(ltrain, k) for k in (1, 2, 3, 5, 10))
    accel = g1 - g5
    level = ltrain[-1]
    below_peak = ltrain[-1] - ltrain[peak]

    f = {
        "g1": g1,
        "g2": g2,
        "g3": g3,
        "g5": g5,
        "g10": g10,
        "accel": accel,
        "vol": float(np.std(d)) if len(d) > 1 else 0.0,
        "level": level,
        # Scaled so the square does not dwarf every other column: shares run
        # from about e^-14 to e^-3, so `level` alone is a large number.
        "level2": (level / 10.0) ** 2,
        "below_peak": below_peak,
        "yrs_since_peak": min(n - 1 - peak, 60) / 60.0,
        "age": min(n, 145) / 145.0,
        "yrs_since_trough": min(len(ltrain[-30:]) - 1 - trough, 30) / 30.0,
    }
    if "inter" in sets:
        s = level / 10.0
        f |= {
            "lvl_g1": s * g1,
            "lvl_g3": s * g3,
            "lvl_g5": s * g5,
            "lvl_g10": s * g10,
            "lvl_accel": s * accel,
            "lvl_peak": s * below_peak,
        }
    if "cohort" in sets:
        end, init, sexc = (np.log(np.maximum(c, FLOOR)) for c in cohort_hist)
        f |= {
            "ce_g3": _growth(end, 3),
            "ce_g10": _growth(end, 10),
            "ci_g5": _growth(init, 5),
            "cs_g5": _growth(sexc, 5),
            "rel_g3": g3 - _growth(end, 3),
            "ce_level": end[-1],
            "own_frac": level - end[-1],
        }
    return f


def feat_names(sets):
    names = list(BASE_FEATS)
    if "inter" in sets:
        names += INTER_FEATS
    if "cohort" in sets:
        names += COHORT_FEATS
    return names


def rows_for(series, origins, sets, coh, min_train=10, extra=None):
    """`extra`, if given, appends features to every row (see `pooled3.Lifecycle`).

    It must expose `.names` and be callable as `extra(key, years, vals, ltr)`
    on the training slice; the base feature block is unchanged either way.
    """
    names = feat_names(sets) + (list(extra.names) if extra is not None else [])
    out = []
    for key, years, vals, rank in series:
        tr_all = years <= max(origins)
        for o in origins:
            tr = years <= o
            if tr.sum() < min_train or years[tr][-1] != o:
                continue
            ltr = np.log(np.maximum(vals[tr], FLOOR))
            cohort_hist = None
            if "cohort" in sets:
                own = np.zeros(o - 1880 + 1)
                idx = years[tr_all] - 1880
                keep = years[tr_all] <= o
                own[idx[keep]] = vals[tr_all][keep]
                cohort_hist = [np.maximum(c - own, 0.0) for c in coh.series(key, o)]
            f = features(ltr, cohort_hist, sets)
            if extra is not None:
                f |= extra(key, years[tr], vals[tr], ltr)
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
                    "x": np.array([f[k] for k in names]),
                    "y": tgt,
                    "last": float(vals[tr][-1]),
                    "actual": [fut.get(o + i) for i in range(1, H + 1)],
                    "level": float(ltr[-1]),
                }
            )
    return out


def fit_ridge(X, y, lam, weights=None):
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Z = np.hstack([np.ones((len(X), 1)), (X - mu) / sd])
    if weights is None:
        A, b = Z.T @ Z, Z.T @ y
    else:
        ZW = Z * weights[:, None]
        A, b = Z.T @ ZW, ZW.T @ y
    A = A + lam * np.eye(Z.shape[1])
    A[0, 0] -= lam
    return (np.linalg.solve(A, b), mu, sd)


def predict(model, X):
    w, mu, sd = model
    return np.hstack([np.ones((len(X), 1)), (X - mu) / sd]) @ w


_TRAIN_CACHE = {}


def train_rows(series, origin, sets, coh, extra=None):
    """Training rows for one origin, built once and reused across penalties."""
    key = (id(series), origin, tuple(sorted(sets)), getattr(extra, "tag", None))
    if key not in _TRAIN_CACHE:
        _TRAIN_CACHE[key] = rows_for(
            series, list(range(1930, origin - H + 1)), sets, coh, extra=extra
        )
    return _TRAIN_CACHE[key]


def evict_train_rows(origin=None):
    """Drop cached training rows for one origin, or all of them.

    Each origin's rows are ~640k dicts, on the order of a gigabyte. Caching
    them across a penalty or hyperparameter sweep at one origin is the point;
    holding twenty-five origins at once is an out-of-memory kill.
    """
    for key in [k for k in _TRAIN_CACHE if origin is None or k[1] == origin]:
        del _TRAIN_CACHE[key]


def pop_weights(rows, power=1.0, clip=50.0):
    """Popularity weights: the fit should care most about names people look up.

    exp(level) is the name's share of births; the exponent sets how hard the
    loss leans on the popular ones and the clip stops the handful of giants
    from becoming the whole fit.
    """
    w = np.exp(np.array([r["level"] for r in rows])) ** power
    return np.clip(w / w.mean(), 0.0, clip)


def train(series, origin, sets, coh, lam, weight, power=1.0, clip=50.0, extra=None):
    tr = train_rows(series, origin, sets, coh, extra)
    X = np.vstack([r["x"] for r in tr])
    Y = np.vstack([r["y"] for r in tr])
    w = pop_weights(tr, power, clip) if weight == "pop" else None
    models = []
    for i in range(H):
        ok = ~np.isnan(Y[:, i])
        models.append(fit_ridge(X[ok], Y[ok, i], lam, None if w is None else w[ok]))
    return models, len(X)


def forecast(models, rows):
    X = np.vstack([r["x"] for r in rows])
    P = np.column_stack([predict(models[i], X) for i in range(H)])
    return [np.exp(p) * r["last"] for r, p in zip(rows, P, strict=True)]


def evaluate(rows, preds, tiers=("top100", "top1000", "top5000", "rest")):
    """Skill against the naive baseline, by popularity tier."""
    res = {}
    for tier in tiers:
        num = den = 0.0
        per = []
        for r, p in zip(rows, preds, strict=True):
            if any(v is None for v in r["actual"]) or bucket(r["rank"]) != tier:
                continue
            a = np.array(r["actual"], dtype=float)
            e_m = np.abs(p - a)
            e_n = np.abs(r["last"] - a)
            num += e_m.sum()
            den += e_n.sum()
            per.append(1 - e_m.mean() / e_n.mean() if e_n.mean() > 0 else 0.0)
        if per:
            res[tier] = {
                "n": len(per),
                "pool": 1 - num / den,
                "med": float(np.median(per)),
                "beat": float(np.mean(np.array(per) > 0.001)),
            }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="inter,cohort", help="comma list of: inter, cohort")
    ap.add_argument("--lam", type=float, default=None, help="ridge penalty; omit to tune")
    ap.add_argument("--lams", default="3,10,30,100,300,1000")
    ap.add_argument("--weight", default="none", choices=["none", "pop"])
    ap.add_argument("--power", type=float, default=1.0, help="popularity weight exponent")
    ap.add_argument("--clip", type=float, default=50.0, help="cap on a single row's weight")
    ap.add_argument("--train-pool", default="all", choices=["all", "top1000"])
    ap.add_argument("--tune-origin", type=int, default=2014)
    ap.add_argument("--eval-origins", default="2014,2019")
    ap.add_argument("--name", default="pooled2")
    ap.add_argument("--top", type=int, default=None)
    ap.add_argument("--mid", type=int, default=0)
    ap.add_argument("--rest", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    sets = {s for s in a.sets.split(",") if s}
    coh = Cohorts() if "cohort" in sets else None
    all_series = load_all()
    if a.train_pool == "top1000":
        all_series = [s for s in all_series if s[3] <= 1000]
    eval_series = load(a.top, a.mid, a.rest)

    lam = a.lam
    if lam is None:
        # Chosen on the tuning origin only; the later origin stays untouched.
        rows = rows_for(eval_series, [a.tune_origin], sets, coh)
        best = (-np.inf, None)
        for cand in [float(x) for x in a.lams.split(",")]:
            models, _ = train(all_series, a.tune_origin, sets, coh, cand, a.weight, a.power, a.clip)
            sc = evaluate(rows, forecast(models, rows), tiers=("top100", "top1000"))
            score = np.mean([sc[t]["pool"] for t in sc])
            print(f"  lam={cand:7.1f}  mean poolSkill(top100,101-1000)={score:.4f}", flush=True)
            if score > best[0]:
                best = (score, cand)
        lam = best[1]
        print(f"  chosen lam={lam}", flush=True)

    out_rows = []
    for eo in [int(x) for x in a.eval_origins.split(",")]:
        models, n_train = train(all_series, eo, sets, coh, lam, a.weight, a.power, a.clip)
        rows = rows_for(eval_series, [eo], sets, coh)
        preds = forecast(models, rows)
        sc = evaluate(rows, preds)
        head = (
            f"{a.name} sets={sorted(sets)} lam={lam} weight={a.weight}"
            f"^{a.power} clip={a.clip} pool={a.train_pool}"
        )
        print(f"\n{head}\norigin {eo} (trained on {n_train:,} name-origins)")
        for tier, s in sc.items():
            print(
                f"  {tier:8} n={s['n']:5d}  pool={s['pool']:+.3f}  "
                f"med={s['med']:+.3f}  beat={100 * s['beat']:.1f}%"
            )
        for r, p in zip(rows, preds, strict=True):
            if any(v is None for v in r["actual"]):
                continue
            out_rows.append(
                {
                    "key": r["key"],
                    "rank": r["rank"],
                    "origin": r["origin"],
                    "method": a.name,
                    "secs": 0.0,
                    "pred": list(map(float, p)),
                    "actual": r["actual"],
                    "last": r["last"],
                    "years": [r["origin"] + i for i in range(1, H + 1)],
                }
            )
        top = sorted(
            zip(feat_names(sets), models[-1][0][1:], strict=True), key=lambda t: -abs(t[1])
        )[:8]
        print("  h5 coef: " + ", ".join(f"{k}={v:+.3f}" for k, v in top))

    if a.out:
        with open(a.out, "w") as fh:
            for r in out_rows:
                fh.write(json.dumps(r) + "\n")
        print(f"\nwrote {len(out_rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
