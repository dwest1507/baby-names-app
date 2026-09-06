"""Pooled forecaster, third generation: gradient-boosted trees instead of a ridge.

`pooled2.py` is linear in hand-built features, and the single biggest gain found
in round 2 came from adding *one* interaction family by hand. That is the
signature of a response surface the linear model cannot reach, so this fits the
same problem with boosted trees, which find interactions themselves.

Everything else is held fixed on purpose so the comparison is like-for-like:
the same feature builder (`pooled2.features`), the same target
(`log(y_{t+h} / y_t)`, one model per horizon), the same `share^0.5` popularity
weights, the same train/test origin discipline — training rows are name-origins
whose five-year targets had already been observed at the origin being forecast,
and every hyperparameter is chosen on earlier origins than the one scored.

`--sets life` adds the lifecycle features issue #34 asks for: how fast the name
rose, how long it has been falling, whether it has flattened out, and what the
same name is doing in the other sex.

    pooled3.py --model gbt --sets inter,life --weight pop --power 0.5 \\
               --tune-origins 2009,2014 --eval-origins 2019 --name gbt_life
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
from cohorts import Cohorts  # noqa: E402

from data import load, load_all  # noqa: E402

H = pooled2.H
FLOOR = pooled2.FLOOR

LIFE_FEATS = [
    "up_slope",
    "rise_yrs",
    "fall_slope",
    "slope_ratio",
    "half_peak_yrs",
    "vol_ratio",
    "flat_yrs",
    "x_present",
    "x_level",
    "x_g3",
    "x_g5",
    "x_frac",
]

# Grids are small on purpose: every candidate is fitted on ~640k rows x 5
# horizons at each tuning origin, and the point of the sweep is to show the
# result is not knife-edge, not to squeeze the last 0.001 out of it.
GRIDS = {
    "leaves": [31, 63, 127, 255],
    "lr": [0.03, 0.05, 0.1],
    "trees": [300, 600, 1000],
    "min_child": [50, 200, 1000],
}


class Lifecycle:
    """Where in its rise-and-fall a name is, plus what the other sex is doing.

    `below_peak` and `yrs_since_peak` in `pooled2` are crude proxies for this.
    The cross-sex block matters because a unisex name's two series move
    together, and the model otherwise sees each half in isolation.
    """

    tag = "life"
    names = LIFE_FEATS

    def __init__(self, series):
        # key is "name|sex"; the counterpart is the same name, other sex.
        self._by_key = {str(k): (y, v) for k, y, v, _ in series}

    def _counterpart(self, key, origin):
        name, sex = str(key).rsplit("|", 1)
        other = self._by_key.get(f"{name}|{'F' if sex == 'M' else 'M'}")
        if other is None:
            return None
        years, vals = other
        m = years <= origin
        if m.sum() < 4:
            return None
        return np.log(np.maximum(vals[m], FLOOR))

    def __call__(self, key, years, vals, ltr):
        n = len(ltr)
        peak = int(np.argmax(ltr))
        # The trough *before* the peak, so `up_slope` measures the actual climb
        # rather than a decline that happens to precede a lower later high.
        pre = int(np.argmin(ltr[: peak + 1])) if peak > 0 else 0
        rise_yrs = peak - pre
        fall_yrs = n - 1 - peak
        up_slope = (ltr[peak] - ltr[pre]) / rise_yrs if rise_yrs > 0 else 0.0
        fall_slope = (ltr[-1] - ltr[peak]) / fall_yrs if fall_yrs > 0 else 0.0
        g5 = pooled2._growth(ltr, 5)

        # Half the peak *share*, i.e. log(peak) - log(2): the point at which a
        # name has given back half of what it ever had.
        half = ltr[peak] - np.log(2.0)
        above = np.where(ltr >= half)[0]
        half_peak_yrs = (n - 1 - int(above[-1])) if len(above) else n - 1

        d = np.diff(ltr)
        v_recent = float(np.std(d[-5:])) if len(d) >= 5 else 0.0
        v_hist = float(np.std(d[-20:])) if len(d) >= 6 else 0.0
        flat = 0
        for step in d[::-1]:
            if abs(step) >= 0.05:
                break
            flat += 1

        x = self._counterpart(key, int(years[-1]))
        if x is None:
            x_present, x_level, x_g3, x_g5, x_frac = 0.0, -16.0, 0.0, 0.0, 1.0
        else:
            x_present = 1.0
            x_level = float(x[-1])
            x_g3, x_g5 = pooled2._growth(x, 3), pooled2._growth(x, 5)
            own = np.exp(ltr[-1])
            x_frac = float(own / (own + np.exp(x_level) + 1e-300))

        return {
            "up_slope": up_slope,
            "rise_yrs": min(rise_yrs, 60) / 60.0,
            "fall_slope": fall_slope,
            # Is the name still moving the way it moved on the way up, and how
            # fast? Clipped because the denominator can be tiny.
            "slope_ratio": float(np.clip(g5 / (abs(up_slope) + 1e-3), -10.0, 10.0)),
            "half_peak_yrs": min(half_peak_yrs, 60) / 60.0,
            "vol_ratio": float(np.clip(v_recent / (v_hist + 1e-6), 0.0, 10.0)),
            "flat_yrs": min(flat, 20) / 20.0,
            "x_present": x_present,
            "x_level": x_level,
            "x_g3": x_g3,
            "x_g5": x_g5,
            "x_frac": x_frac,
        }


def parse_origins(spec):
    """`2009,2014`, or `1995:2019`, or `1995:2019:5` — inclusive of both ends."""
    out = []
    for part in filter(None, spec.split(",")):
        if ":" in part:
            bits = [int(x) for x in part.split(":")]
            out.extend(range(bits[0], bits[1] + 1, bits[2] if len(bits) > 2 else 1))
        else:
            out.append(int(part))
    return sorted(dict.fromkeys(out))


def _lgb_params(hp, objective, seed):
    p = {
        "objective": objective,
        "num_leaves": hp["leaves"],
        "learning_rate": hp["lr"],
        "n_estimators": hp["trees"],
        "min_child_samples": hp["min_child"],
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "max_bin": 255,
        "n_jobs": -1,
        "random_state": seed,
        "verbosity": -1,
        "force_col_wise": True,
    }
    if objective == "huber":
        p["alpha"] = 1.0
    return p


def train_gbt(series, origin, sets, coh, hp, weight, power, clip, extra, objective, seed=0):
    """One booster per horizon, on the rows observable at `origin`."""
    import lightgbm as lgb

    tr = pooled2.train_rows(series, origin, sets, coh, extra)
    X = np.vstack([r["x"] for r in tr])
    Y = np.vstack([r["y"] for r in tr])
    w = pooled2.pop_weights(tr, power, clip) if weight == "pop" else None
    models = []
    for i in range(H):
        ok = ~np.isnan(Y[:, i])
        m = lgb.LGBMRegressor(**_lgb_params(hp, objective, seed))
        m.fit(X[ok], Y[ok, i], sample_weight=None if w is None else w[ok])
        models.append(m)
    return models, len(X)


def forecast_gbt(models, rows):
    X = np.vstack([r["x"] for r in rows])
    P = np.column_stack([m.predict(X) for m in models])
    return [np.exp(p) * r["last"] for r, p in zip(rows, P, strict=True)]


def score_of(sc, tiers=("top100", "top1000")):
    """One number to tune on: the tiers visitors actually look at."""
    return float(np.mean([sc[t]["pool"] for t in tiers if t in sc]))


def show(sc, label):
    print(f"  {label}", flush=True)
    for tier, s in sc.items():
        print(
            f"    {tier:8} n={s['n']:5d}  pool={s['pool']:+.3f}  "
            f"med={s['med']:+.3f}  beat={100 * s['beat']:.1f}%",
            flush=True,
        )


def tune(all_series, eval_series, origins, sets, coh, base, a, extra):
    """Coordinate descent over the grid, scored only on origins < the test one."""
    hp = dict(base)
    rows = {o: pooled2.rows_for(eval_series, [o], sets, coh, extra=extra) for o in origins}

    def run(cand):
        s = []
        for o in origins:
            models, _ = train_gbt(
                all_series, o, sets, coh, cand, a.weight, a.power, a.clip, extra, a.objective
            )
            s.append(score_of(pooled2.evaluate(rows[o], forecast_gbt(models, rows[o]))))
        return float(np.mean(s))

    best = run(hp)
    print(f"  start {hp} -> {best:.4f}", flush=True)
    for param, values in GRIDS.items():
        for v in values:
            if v == hp[param]:
                continue
            cand = dict(hp, **{param: v})
            sc = run(cand)
            print(f"  {param}={v:<6} {sc:.4f}{'  *' if sc > best else ''}", flush=True)
            if sc > best:
                best, hp = sc, cand
    print(f"  chosen {hp} -> {best:.4f}", flush=True)
    return hp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gbt", choices=["gbt", "ridge"])
    ap.add_argument("--sets", default="inter", help="comma list of: inter, cohort, life")
    ap.add_argument("--objective", default="l2", choices=["l2", "l1", "huber", "quantile"])
    ap.add_argument("--weight", default="pop", choices=["none", "pop"])
    ap.add_argument("--power", type=float, default=0.5)
    ap.add_argument("--clip", type=float, default=50.0)
    ap.add_argument("--lam", type=float, default=100.0, help="ridge penalty, --model ridge only")
    ap.add_argument("--leaves", type=int, default=63)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--trees", type=int, default=600)
    ap.add_argument("--min-child", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=1, help="average this many seeds' forecasts")
    ap.add_argument("--tune", action="store_true", help="sweep hyperparameters first")
    ap.add_argument(
        "--grid",
        default="",
        help="override the sweep, e.g. 'leaves=7,15,31;lr=0.01,0.03;trees=150,300'",
    )
    ap.add_argument("--tune-origins", default="2009,2014")
    ap.add_argument("--eval-origins", default="2019")
    ap.add_argument("--name", default="pooled3")
    ap.add_argument("--top", type=int, default=100000)
    ap.add_argument("--mid", type=int, default=1200)
    ap.add_argument("--rest", type=int, default=1200)
    ap.add_argument("--importance", action="store_true", help="print h5 gain importances")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    sets = {s for s in a.sets.split(",") if s}
    coh = Cohorts() if "cohort" in sets else None
    all_series = load_all()
    extra = Lifecycle(all_series) if "life" in sets else None
    sets.discard("life")
    eval_series = load(a.top, a.mid, a.rest)

    for part in filter(None, a.grid.split(";")):
        k, v = part.split("=")
        GRIDS[k] = [float(x) if k == "lr" else int(x) for x in v.split(",")]

    hp = {"leaves": a.leaves, "lr": a.lr, "trees": a.trees, "min_child": a.min_child}
    if a.tune and a.model == "gbt":
        hp = tune(
            all_series,
            eval_series,
            parse_origins(a.tune_origins),
            sets,
            coh,
            hp,
            a,
            extra,
        )

    out_rows = []
    for eo in parse_origins(a.eval_origins):
        # One origin's training rows at a time: see pooled2.evict_train_rows.
        pooled2.evict_train_rows()
        t0 = time.time()
        rows = pooled2.rows_for(eval_series, [eo], sets, coh, extra=extra)
        if a.model == "ridge":
            models, n_train = pooled2.train(
                all_series, eo, sets, coh, a.lam, a.weight, a.power, a.clip, extra
            )
            preds = pooled2.forecast(models, rows)
        else:
            # Averaging seeds in log space: boosting with subsampling is a
            # stochastic fit, and one seed's noise is not a property of the
            # method.
            acc = np.zeros((len(rows), H))
            for s in range(a.seeds):
                models, n_train = train_gbt(
                    all_series, eo, sets, coh, hp, a.weight, a.power, a.clip, extra, a.objective, s
                )
                acc += np.log(np.vstack(forecast_gbt(models, rows)))
            preds = list(np.exp(acc / a.seeds))
        fit_s = time.time() - t0

        head = f"{a.name} model={a.model} sets={sorted(sets) + (['life'] if extra else [])}"
        if a.model == "gbt":
            head += f" {hp} obj={a.objective} seeds={a.seeds}"
        else:
            head += f" lam={a.lam}"
        print(f"\n{head}\norigin {eo} ({n_train:,} training name-origins, {fit_s:.0f}s)")
        show(pooled2.evaluate(rows, preds), "")

        if a.importance and a.model == "gbt":
            fn = pooled2.feat_names(sets) + (list(extra.names) if extra else [])
            imp = models[-1].booster_.feature_importance("gain")
            top = sorted(zip(fn, imp / imp.sum(), strict=True), key=lambda t: -t[1])[:12]
            print("  h5 gain: " + ", ".join(f"{k}={100 * v:.1f}%" for k, v in top))

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

    if a.out:
        with open(a.out, "w") as fh:
            for r in out_rows:
                fh.write(json.dumps(r) + "\n")
        print(f"\nwrote {len(out_rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
