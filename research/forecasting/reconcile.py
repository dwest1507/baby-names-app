"""Cross-sectional reconciliation: shares have to add up, so make them add up.

Issue #34's second recommendation. `popularity_percent` is a share of one
year's births *within a sex*, so across all names of a sex in a year it sums to
1 by construction. Every forecast in this harness is made one name at a time
and nothing ties them together; the sum of the forecasts is free to drift away
from what the sum has to be. A per-name metric cannot see that error at all.

The eligible set (names still in use in 2024, ADR 0001) covers 92-98% of each
year's births and its total moves by at most ~1% over five years, so the
adding-up target is both real and easy to forecast. Three ways to set it:

    naive    the total stays where it was at the origin
    drift    damped log-linear extrapolation of the total's own history
    oracle   the total actually observed — not a usable forecast, it is the
             bound that says how much of any gain comes from the constraint
             rather than from knowing the answer

and three ways to spread the discrepancy across names:

    prop     one multiplicative factor for everyone; equivalently, a constant
             shift in log space, so relative ordering is untouched
    ols      the textbook equal-weight reconciliation: every name absorbs the
             same absolute amount, clipped at zero
    vol      a log shift proportional to each name's own recent volatility, so
             the names whose futures are least certain absorb most of it

    reconcile.py .work/gbt_full.jsonl --out .work/recon.jsonl
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SP)
from data import SERIES, bucket, load_all  # noqa: E402

H = 5
FLOOR = 1e-12
TIERS = ("top100", "top1000", "top5000", "rest")


def sex_of(key):
    return str(key).rsplit("|", 1)[1]


class Panel:
    """Every series as one dense name x year matrix of shares.

    The constraint set has to be *the names actually being forecast at this
    origin* — a name whose five-year window is incomplete is dropped from the
    forecast file, and reconciling to a total that includes it would build a
    known shortfall into every factor. So the totals are summed over whatever
    keys the caller passes in, not over the corpus.
    """

    def __init__(self, series):
        self.years = np.arange(1880, 2025)
        self.index = {}
        M = np.zeros((len(series), len(self.years)))
        for i, (key, years, vals, _rank) in enumerate(series):
            self.index[str(key)] = i
            M[i, np.asarray(years) - self.years[0]] = vals
        self.M = M

    def totals(self, keys):
        """The summed share of `keys`, per year, oldest first."""
        idx = [self.index[k] for k in keys if k in self.index]
        return self.M[idx].sum(axis=0)

    def at(self, totals, year):
        return float(totals[year - self.years[0]])


def target_totals(panel, totals, origin, scheme, k=10, phi=0.8):
    """The five annual totals to reconcile to, under one of the three schemes."""
    s0 = panel.at(totals, origin)
    if scheme == "oracle":
        return np.array([panel.at(totals, origin + h) for h in range(1, H + 1)])
    if scheme == "naive":
        return np.full(H, s0)
    lh = np.log(np.maximum([panel.at(totals, y) for y in range(origin - k, origin + 1)], FLOOR))
    slope = (lh[-1] - lh[0]) / (len(lh) - 1)
    return np.array(
        [s0 * np.exp(slope * sum(phi**j for j in range(1, h + 1))) for h in range(1, H + 1)]
    )


def volatilities(keys_needed):
    d = np.load(SERIES, allow_pickle=True)
    keys, years, vals = d["keys"], d["years"], d["vals"]
    kidx = {k: i for i, k in enumerate(keys)}
    out = {}
    for key, origin in keys_needed:
        i = kidx[key]
        lv = np.log(np.maximum(vals[i][years[i] <= origin], FLOOR))
        out[(key, origin)] = float(np.std(np.diff(lv[-11:]))) if len(lv) > 2 else 0.2
    return out


def solve_vol_shift(p, v, target):
    """c such that sum(p * exp(c*v)) == target. Monotone in c, so bisect."""
    lo, hi = -5.0, 5.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if float((p * np.exp(mid * v)).sum()) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def reconcile_group(preds, vols, target, how):
    """One (origin, sex, horizon) slice: nudge `preds` so they sum to `target`."""
    p = np.asarray(preds, dtype=float)
    s = p.sum()
    if s <= 0:
        return p
    if how == "prop":
        return p * (target / s)
    if how == "ols":
        return np.maximum(p + (target - s) / len(p), 0.0)
    return p * np.exp(solve_vol_shift(p, np.asarray(vols), target) * np.asarray(vols))


def skill_table(rows, key):
    """poolSkill and the per-name median, per tier, reading forecasts out of `key`."""
    acc = defaultdict(lambda: np.zeros(2))
    per = defaultdict(list)
    for r in rows:
        a = np.array(r["actual"], dtype=float)
        t = bucket(r["rank"])
        e_m = np.abs(np.asarray(r[key]) - a)
        e_n = np.abs(r["last"] - a)
        acc[t] += (e_m.sum(), e_n.sum())
        per[t].append(1 - e_m.mean() / e_n.mean() if e_n.mean() > 0 else 0.0)
    return {
        t: (1 - acc[t][0] / acc[t][1], float(np.median(per[t]))) for t in TIERS if acc[t][1] > 0
    }


def bootstrap(rows, field, tier, draws=5000, seed=0):
    """Cluster bootstrap over names of the reconciled-minus-free poolSkill gap."""
    units = defaultdict(lambda: np.zeros(3))
    for r in rows:
        if bucket(r["rank"]) != tier:
            continue
        a = np.array(r["actual"], dtype=float)
        units[r["key"]] += (
            np.abs(np.asarray(r[field]) - a).sum(),
            np.abs(np.asarray(r["pred"]) - a).sum(),
            np.abs(r["last"] - a).sum(),
        )
    if not units:
        return None
    er, ef, en = np.array(list(units.values())).T
    idx = np.random.default_rng(seed).integers(0, len(er), size=(draws, len(er)))
    diff = (1 - er[idx].sum(1) / en[idx].sum(1)) - (1 - ef[idx].sum(1) / en[idx].sum(1))
    return len(units), diff.mean(), *np.percentile(diff, [2.5, 97.5]), (diff > 0).mean()


def origin_wins(rows, field):
    """How many origins the reconciled arm beats the free one at, per tier."""
    acc = defaultdict(lambda: np.zeros(3))
    for r in rows:
        a = np.array(r["actual"], dtype=float)
        acc[(r["origin"], bucket(r["rank"]))] += (
            np.abs(np.asarray(r[field]) - a).sum(),
            np.abs(np.asarray(r["pred"]) - a).sum(),
            np.abs(r["last"] - a).sum(),
        )
    out = {}
    for t in TIERS:
        origins = sorted({o for (o, tt) in acc if tt == t})
        out[t] = (sum(acc[(o, t)][0] < acc[(o, t)][1] for o in origins), len(origins))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="forecasts over the whole eligible set")
    ap.add_argument(
        "--method", default=None, help="which method in the file (default: the only one)"
    )
    ap.add_argument("--targets", default="naive,drift,oracle")
    ap.add_argument("--hows", default="prop,ols,vol")
    ap.add_argument("--set", default="all", choices=["all", "top1000"], help="the constraint set")
    ap.add_argument("--phi", type=float, default=0.8, help="damping for the drift target")
    ap.add_argument("--window", type=int, default=10, help="years of total history for the drift")
    ap.add_argument("--diagnose", action="store_true", help="print the aggregate drift table only")
    ap.add_argument(
        "--test",
        default=None,
        help="one arm, e.g. 'naive_prop' — bootstrap its gap over the free forecast and "
        "count the origins it wins",
    )
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rows = [json.loads(line) for line in open(a.path)]
    methods = sorted({r["method"] for r in rows})
    method = a.method or methods[0]
    rows = [r for r in rows if r["method"] == method]
    rank_max = 1000 if a.set == "top1000" else None
    if rank_max:
        rows = [r for r in rows if r["rank"] <= rank_max]

    series = load_all()
    panel = Panel(series)
    origins = sorted({r["origin"] for r in rows})
    print(f"method={method}  constraint set={a.set}  origins={origins[0]}-{origins[-1]}")

    groups = defaultdict(list)
    for r in rows:
        groups[(r["origin"], sex_of(r["key"]))].append(r)
    # One total series per (origin, sex), over exactly that group's names.
    totals = {g: panel.totals([r["key"] for r in grp]) for g, grp in groups.items()}

    # How far does the free forecast's sum wander from the sum it has to hit?
    print("\n=== aggregate drift: sum(forecast) / sum(actual), by horizon ===")
    print(
        f"{'origin':>7} {'sex':>4} {'names':>6} "
        + " ".join(f"{'h' + str(h):>7}" for h in range(1, H + 1))
        + f" {'naiveH5':>8} {'ofBirths':>8}"
    )
    shown = set(origins[:: max(len(origins) // 8, 1)])
    for (o, sx), grp in sorted(groups.items()):
        P = np.sum([r["pred"] for r in grp], axis=0)
        A = np.sum([r["actual"] for r in grp], axis=0)
        L = float(np.sum([r["last"] for r in grp]))
        if o in shown:
            print(
                f"{o:7d} {sx:>4} {len(grp):6d} "
                + " ".join(f"{P[h] / A[h]:7.4f}" for h in range(H))
                + f" {L / A[4]:8.4f}"
                # `popularity_percent` sums to 1 over every name of a sex in a
                # year, so the group's own total *is* its share of births.
                + f" {panel.at(totals[(o, sx)], o):8.3f}"
            )
    allP = np.zeros(H)
    allA = np.zeros(H)
    for grp in groups.values():
        allP += np.sum([r["pred"] for r in grp], axis=0)
        allA += np.sum([r["actual"] for r in grp], axis=0)
    print(
        "pooled over every origin and sex: "
        + " ".join(f"h{h + 1}={allP[h] / allA[h]:.4f}" for h in range(H))
    )
    if a.diagnose:
        return

    vols = volatilities({(r["key"], r["origin"]) for r in rows})

    def line(label, sc):
        cells = []
        for t in TIERS:
            pool, med = sc.get(t, (float("nan"), float("nan")))
            cells.append(f"{pool:8.3f} {med:7.3f}")
        print(f"{label:24} " + " ".join(cells))

    print("\n=== skill after reconciliation (poolSkill / medSkill) ===")
    print(f"{'arm':24} " + " ".join(f"{t:>16}" for t in TIERS))
    line(f"{method} (free)", skill_table(rows, "pred"))

    out_rows = []
    for scheme in a.targets.split(","):
        for how in a.hows.split(","):
            field = f"rec_{scheme}_{how}"
            for (o, sx), grp in groups.items():
                tgt = target_totals(panel, totals[(o, sx)], o, scheme, a.window, a.phi)
                P = np.array([r["pred"] for r in grp])
                V = np.array([vols[(r["key"], r["origin"])] for r in grp])
                R = np.column_stack([reconcile_group(P[:, h], V, tgt[h], how) for h in range(H)])
                for r, rec in zip(grp, R, strict=True):
                    r[field] = list(map(float, rec))
            name = f"{method}_{scheme}_{how}"
            line(name, skill_table(rows, field))
            if a.out:
                out_rows.append((name, field))

    if a.test:
        field = f"rec_{a.test}"
        if field not in rows[0]:
            print(f"\n{a.test} was not built; pass it via --targets/--hows")
            return
        wins = origin_wins(rows, field)
        print(f"\n=== {method}_{a.test} minus {method}, cluster bootstrap over names ===")
        print(f"{'tier':10} {'names':>6} {'diff [2.5%, 97.5%]':>30} {'origins won':>13}")
        for t in TIERS:
            b = bootstrap(rows, field, t)
            if b is None:
                continue
            n, mean, lo_ci, hi_ci, p = b
            w, tot = wins[t]
            print(
                f"{t:10} {n:6d} {mean:+9.4f} [{lo_ci:+.4f}, {hi_ci:+.4f}]  P={100 * p:3.0f}% "
                f"{w:6d} / {tot}"
            )

    if a.out:
        # The free forecast and a naive arm go in too, so `paired.py` can
        # bootstrap the reconciled-minus-free gap out of this one file.
        arms = [(method, "pred"), ("naive", None)] + out_rows
        with open(a.out, "w") as fh:
            for name, field in arms:
                for r in rows:
                    fh.write(
                        json.dumps(
                            {
                                "key": r["key"],
                                "rank": r["rank"],
                                "origin": r["origin"],
                                "method": name,
                                "secs": 0.0,
                                "pred": [r["last"]] * H if field is None else r[field],
                                "actual": r["actual"],
                                "last": r["last"],
                                "years": r["years"],
                            }
                        )
                        + "\n"
                    )
        print(f"\nwrote {len(arms) * len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
