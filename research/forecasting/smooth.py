"""Is the five-year path a trajectory, or five independent guesses in a row?

Issue #34's fifth recommendation. `pooled3.py` fits one booster per horizon, and
nothing ties them together: h3's model has never seen h2's output. A visitor
reads the *shape* of the dashed line, so a path that rises, dips and rises again
is a claim the model is not making on purpose. Round 5 checked the equivalent
question for the bands (0.66% narrow with the horizon); this checks the centre.

Two questions, in order:

1. How jagged is the predicted path, in units a reader would notice — direction
   reversals, and how big the reversal is against the size of the whole move.
   Actual paths are the reference: reality is jagged, and a conditional mean
   should be *smoother* than a draw from it, never rougher.
2. What does it cost to smooth it? Three smoothers, scored on the same
   poolSkill as everything else. A presentation fix that is free on accuracy is
   shippable; one that costs accuracy is a trade to be decided, not assumed.

    smooth.py .work/gbt_full.jsonl --smooth-score
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

FLOOR = 1e-12
H = 5
TIERS = ("top100", "top1000", "top5000", "rest")
EPS = 0.02  # a step smaller than 2% in log share is not a visible direction


def steps(path, last):
    """Year-over-year log steps of a five-point path, starting from `last`."""
    lp = np.log(np.maximum(np.asarray(path, dtype=float), FLOOR))
    return np.diff(np.concatenate([[np.log(max(last, FLOOR))], lp]))


def path_stats(d):
    """Shape descriptors for one path's step vector."""
    sign = np.where(np.abs(d) < EPS, 0, np.sign(d))
    nz = sign[sign != 0]
    reversals = int((np.diff(nz) != 0).sum()) if len(nz) > 1 else 0
    total = float(np.abs(d.sum()))
    # How much of the movement is walked back: the sum of the moves that go
    # against the net direction, as a share of the net move.
    net = np.sign(d.sum()) if d.sum() != 0 else 1.0
    against = float(np.abs(d[np.sign(d) != net]).sum())
    return {
        "rev": reversals,
        "any_rev": float(reversals > 0),
        "d2": np.abs(np.diff(d)),
        "backtrack": against / (total + 1e-9),
        "roughness": float(np.abs(np.diff(d)).mean()),
        "move": total,
    }


def describe(rows, key, label):
    """Jaggedness of one path source (model, actuals, or a smoothed variant)."""
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        d = steps(key(r), r["last"])
        s = path_stats(d)
        for t in (bucket(r["rank"]), "ALL"):
            acc[t]["rev"].append(s["any_rev"])
            acc[t]["nrev"].append(s["rev"])
            acc[t]["rough"].append(s["roughness"])
            acc[t]["d2"].append(s["d2"])
            acc[t]["back"].append(min(s["backtrack"], 5.0))
    print(f"\n{label}")
    print(
        f"{'tier':9} {'n':>7} {'reversal%':>10} {'revs/path':>10} {'rough':>8} "
        f"{'d2 p50':>8} {'d2 p95':>8} {'backtrk':>8}"
    )
    for t in TIERS + ("ALL",):
        if t not in acc:
            continue
        a = acc[t]
        d2 = np.concatenate(a["d2"])
        print(
            f"{t:9} {len(a['rev']):7d} {100 * np.mean(a['rev']):10.1f} "
            f"{np.mean(a['nrev']):10.2f} {np.mean(a['rough']):8.4f} "
            f"{np.median(d2):8.4f} {np.quantile(d2, 0.95):8.4f} {np.mean(a['back']):8.3f}"
        )
    return acc


def horizon_bias(rows):
    """Mean step per horizon, model against actual — a shared kink shows here.

    If one horizon's booster is systematically flatter or steeper than its
    neighbours, every path inherits the same bend and the jaggedness is an
    artefact of the fit rather than of any one name.
    """
    m = np.zeros((H, 2))
    a = np.zeros((H, 2))
    for r in rows:
        dm, da = steps(r["pred"], r["last"]), steps(r["actual"], r["last"])
        m[:, 0] += dm
        a[:, 0] += da
        m[:, 1] += 1
        a[:, 1] += 1
    print("\nmean year-over-year log step, by horizon (unweighted over rows)")
    print(f"{'':10} " + " ".join(f"{'h' + str(i + 1):>9}" for i in range(H)))
    print("model      " + " ".join(f"{v:9.4f}" for v in m[:, 0] / m[:, 1]))
    print("actual     " + " ".join(f"{v:9.4f}" for v in a[:, 0] / a[:, 1]))
    print("model/act  " + " ".join(f"{v:9.3f}" for v in (m[:, 0] / m[:, 1]) / (a[:, 0] / a[:, 1])))


def horizon_detail(rows):
    """Per horizon: signed log bias and poolSkill, by tier.

    A shared bend in the path shows up as a bias that is not smooth in `h`.
    """
    bias = defaultdict(lambda: np.zeros((H, 2)))
    err = defaultdict(lambda: np.zeros((H, 2)))
    for r in rows:
        t = bucket(r["rank"])
        lp = np.log(np.maximum(np.asarray(r["pred"], dtype=float), FLOOR))
        la = np.log(np.maximum(np.asarray(r["actual"], dtype=float), FLOOR))
        act = np.array(r["actual"], dtype=float)
        for tier in (t, "ALL"):
            bias[tier][:, 0] += lp - la
            bias[tier][:, 1] += 1
            err[tier][:, 0] += np.abs(np.asarray(r["pred"]) - act)
            err[tier][:, 1] += np.abs(r["last"] - act)
    print("\nmean log bias (model - actual) by horizon")
    print(f"{'tier':9} " + " ".join(f"{'h' + str(i + 1):>9}" for i in range(H)))
    for t in TIERS + ("ALL",):
        b = bias[t]
        print(f"{t:9} " + " ".join(f"{v:9.4f}" for v in b[:, 0] / b[:, 1]))
    print("\npoolSkill by horizon")
    print(f"{'tier':9} " + " ".join(f"{'h' + str(i + 1):>9}" for i in range(H)))
    for t in TIERS + ("ALL",):
        e = err[t]
        print(f"{t:9} " + " ".join(f"{v:9.4f}" for v in 1 - e[:, 0] / e[:, 1]))


def smooth_quad(path, last):
    """Least-squares quadratic in h through the observed last value.

    log(y_h/y_0) = a*h + b*h^2 — the smoothest shape that can still bend once,
    which is what a rise-then-plateau or a decline-then-flatten needs.
    """
    y = np.log(np.maximum(np.asarray(path, dtype=float), FLOOR)) - np.log(max(last, FLOOR))
    hh = np.arange(1, H + 1, dtype=float)
    A = np.column_stack([hh, hh**2])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return np.exp(A @ coef + np.log(max(last, FLOOR)))


def smooth_lin(path, last):
    """Straight line in log space through the origin: no bend at all."""
    y = np.log(np.maximum(np.asarray(path, dtype=float), FLOOR)) - np.log(max(last, FLOOR))
    hh = np.arange(1, H + 1, dtype=float)
    slope = float((hh @ y) / (hh @ hh))
    return np.exp(slope * hh + np.log(max(last, FLOOR)))


def smooth_ma(path, last):
    """Three-point moving average of the steps, re-cumulated.

    Keeps the path's own shape and only takes the corners off, so it is the
    conservative option: it cannot turn a genuine bend into a straight line.
    """
    d = steps(path, last)
    pad = np.concatenate([[d[0]], d, [d[-1]]])
    sm = np.convolve(pad, np.ones(3) / 3, mode="valid")
    return np.exp(np.log(max(last, FLOOR)) + np.cumsum(sm))


def smooth_ma_pin(path, last):
    """`ma`, then rescaled so the five-year endpoint is exactly the raw one.

    Separates the two things `ma` does at once: redistribute the shape, and
    move the level. This arm only redistributes.
    """
    sm = smooth_ma(path, last)
    l0 = np.log(max(last, FLOOR))
    ls = np.log(np.maximum(sm, FLOOR)) - l0
    lr = np.log(np.maximum(np.asarray(path, dtype=float), FLOOR)) - l0
    scale = lr[-1] / ls[-1] if abs(ls[-1]) > 1e-12 else 1.0
    return np.exp(l0 + ls * scale)


def bump_h1(path, last):
    """The raw path with only its first point moved, to `ma`'s h1.

    The horizon table shows h1 is the shrunk one: the model's mean first-year
    step is 48% of the actual against 63-72% at every later horizon, so the
    line starts flatter than it continues. This arm changes nothing else.
    """
    out = list(np.asarray(path, dtype=float))
    out[0] = float(smooth_ma(path, last)[0])
    return np.array(out)


SMOOTHERS = {
    "quad": smooth_quad,
    "lin": smooth_lin,
    "ma": smooth_ma,
    "ma_pin": smooth_ma_pin,
    "h1": bump_h1,
}


def skill_table(rows, variants):
    """poolSkill and medSkill per tier for the raw path and each smoothed one."""
    acc = defaultdict(lambda: defaultdict(lambda: np.zeros(2)))
    per = defaultdict(lambda: defaultdict(list))
    for r in rows:
        act = np.array(r["actual"], dtype=float)
        en = np.abs(r["last"] - act)
        t = bucket(r["rank"])
        for name, p in variants(r).items():
            em = np.abs(np.asarray(p) - act)
            for tier in (t, "ALL"):
                acc[name][tier] += (em.sum(), en.sum())
                per[name][tier].append(1 - em.mean() / en.mean() if en.mean() > 0 else 0.0)
    print(f"\n{'arm':8} " + " ".join(f"{t:>18}" for t in TIERS))
    for name in acc:
        cells = []
        for t in TIERS:
            v = acc[name][t]
            cells.append(f"{1 - v[0] / v[1]:+.4f}/{np.median(per[name][t]):+.4f}")
        print(f"{name:8} " + " ".join(f"{c:>18}" for c in cells))
    print("(poolSkill / medSkill)")


def paired_smoothing(rows, variants, base, other, draws=4000, seed=0):
    """Cluster bootstrap over names of the poolSkill gap `base` - `other`.

    Names, not name-origins: one name's overlapping five-year windows are not
    independent draws (`paired.py`). The origin count is the second reading —
    a pooled gain that comes from three good years is a different claim from
    one that shows up in twenty-two of twenty-five.
    """
    rng = np.random.default_rng(seed)
    print(f"\ncluster bootstrap over names: {base} - {other}")
    print(f"{'tier':9} {'names':>7} {'diff [2.5%, 97.5%]':>32} {'origins won':>13}")
    for tier in TIERS + ("ALL",):
        units = defaultdict(lambda: np.zeros(3))
        per_origin = defaultdict(lambda: np.zeros(3))
        for r in rows:
            if tier != "ALL" and bucket(r["rank"]) != tier:
                continue
            v = variants(r)
            act = np.array(r["actual"], dtype=float)
            cell = (
                np.abs(np.asarray(v[base]) - act).sum(),
                np.abs(np.asarray(v[other]) - act).sum(),
                np.abs(r["last"] - act).sum(),
            )
            units[r["key"]] += cell
            per_origin[r["origin"]] += cell
        if not units:
            continue
        ea, eb, en = np.array(list(units.values())).T
        idx = rng.integers(0, len(ea), size=(draws, len(ea)))
        diff = (1 - ea[idx].sum(1) / en[idx].sum(1)) - (1 - eb[idx].sum(1) / en[idx].sum(1))
        lo, hi = np.percentile(diff, [2.5, 97.5])
        won = sum(v[0] < v[1] for v in per_origin.values())
        print(
            f"{tier:9} {len(units):7d} {diff.mean():+9.4f} [{lo:+.4f}, {hi:+.4f}]"
            f"  P={100 * (diff > 0).mean():.0f}% {won:6d} / {len(per_origin)}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--method", default=None, help="restrict to one method name")
    ap.add_argument("--origins", default=None, help="restrict to these origins")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--smooth-score", action="store_true", help="score the smoothed variants")
    ap.add_argument("--examples", type=int, default=0, help="print the N jaggedest top-100 paths")
    ap.add_argument("--write", default=None, help="write a smoothed copy of the forecasts here")
    ap.add_argument("--with", dest="smoother", default="ma", choices=sorted(SMOOTHERS))
    a = ap.parse_args()

    keep_o = None
    if a.origins:
        from pooled3 import parse_origins

        keep_o = set(parse_origins(a.origins))

    rows = []
    for line in open(a.path):
        r = json.loads(line)
        if a.method and r["method"] != a.method:
            continue
        if keep_o and r["origin"] not in keep_o:
            continue
        if any(v is None for v in r["actual"]):
            continue
        rows.append(r)
        if a.limit and len(rows) >= a.limit:
            break
    print(
        f"{len(rows)} rows, "
        f"origins {min(r['origin'] for r in rows)}-{max(r['origin'] for r in rows)}"
    )

    describe(rows, lambda r: r["pred"], "predicted paths")
    describe(rows, lambda r: r["actual"], "actual paths (the reference: reality is jagged)")
    horizon_bias(rows)
    horizon_detail(rows)

    if a.examples:
        top = [r for r in rows if bucket(r["rank"]) == "top100"]
        top.sort(key=lambda r: -path_stats(steps(r["pred"], r["last"]))["roughness"])
        print("\njaggedest top-100 predicted paths (share x 100, from `last`)")
        for r in top[: a.examples]:
            p = " -> ".join(f"{100 * v:.3f}" for v in [r["last"]] + list(r["pred"]))
            print(f"  {r['key']:16} {r['origin']}  {p}")

    if a.write:
        write_smoothed(rows, a.write, a.smoother)

    if a.smooth_score:

        def variants(r):
            out = {"raw": r["pred"]}
            for name, fn in SMOOTHERS.items():
                out[name] = fn(r["pred"], r["last"])
            return out

        skill_table(rows, variants)
        for name in SMOOTHERS:
            describe(
                rows, lambda r, f=SMOOTHERS[name]: f(r["pred"], r["last"]), f"smoothed: {name}"
            )
        for name in SMOOTHERS:
            paired_smoothing(rows, variants, name, "raw")


def write_smoothed(rows, path, smoother, name_suffix="_sm"):
    """A smoothed copy, so the change can be composed with reconciliation.

    `ma` preserves each path's five-year endpoint exactly but moves h1-h4, so
    it changes the sums the reconciler targets at those horizons. Running the
    two in this order — smooth, then reconcile — is the only order in which
    the reconciled forecasts actually add up.
    """
    fn = SMOOTHERS[smoother]
    with open(path, "w") as fh:
        for r in rows:
            out = dict(r)
            out["method"] = r["method"] + name_suffix
            out["pred"] = [float(v) for v in fn(r["pred"], r["last"])]
            fh.write(json.dumps(out) + "\n")
    print(f"\nwrote {len(rows)} {smoother}-smoothed rows -> {path}")


if __name__ == "__main__":
    main()
