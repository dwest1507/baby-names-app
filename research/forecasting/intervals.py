"""How good is a prediction band, and is fitting the quantiles directly better?

`conformal.py` answers one question — what coverage do residual quantiles
achieve — and answers it for one construction. This scores several
constructions against each other on identical rows, using proper interval
scoring rather than coverage alone, because coverage on its own is trivially
gamed by a wider band.

Arms (all calibrated only on origins earlier than the one they are scored on):

    resid_global    symmetric log-residual quantiles off the point forecast
    resid_tiervol   the same, conditioned on popularity tier x volatility
                    — the band round 3 recommended
    direct          the quantile boosters' own output, uncalibrated
    cqr             direct, then shifted by a conformal correction per horizon
                    (Romano et al.'s conformalised quantile regression)
    cqr_tiervol     the same correction, conditioned on tier x volatility
    cqr_asym        direct, with the two tails corrected separately, so a band
                    that is honest below and short above can be fixed on the
                    side that is actually wrong
    cqr_asym_tier   the same, per popularity tier

Metrics, per popularity tier:

    cover       share of held-out actuals inside the band
    width       mean log width, log(hi/lo) — comparable across arms only at
                equal coverage, which is why the interval score exists
    intScore    Winkler interval score in log space: width plus a miss penalty
                of 2/(1-level) times the distance outside. The proper scoring
                rule for an interval; lower is better, and it is the number to
                read when coverage and width disagree
    asym        mean upper half-width / mean total width. 0.5 is a symmetric
                band; a construction that can see a name is past its peak
                should sit below it
    lo%, hi%    where the misses fall. A band can have exactly the right
                marginal coverage and still be wrong for every name in it, by
                missing low on the ones past their peak and high on the ones
                still rising. Nominal is (1-level)/2 on each side, in every
                conditioning bin

    intervals.py .work/qr_many.jsonl --cal-origins 1995:2009 --test-origins 2010:2019
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SP)
from data import SERIES, bucket  # noqa: E402
from pooled3 import parse_origins  # noqa: E402

FLOOR = 1e-12
H = 5
TIERS = ("top100", "top1000", "top5000", "rest")


def series_context():
    """(key, origin) -> recent log-volatility and how far the name is below its peak."""
    d = np.load(SERIES, allow_pickle=True)
    keys, years, vals = d["keys"], d["years"], d["vals"]
    kidx = {k: i for i, k in enumerate(keys)}

    def ctx(key, origin):
        i = kidx[key]
        y, v = years[i], vals[i]
        lv = np.log(np.maximum(v[y <= origin], FLOOR))
        vol = float(np.std(np.diff(lv[-11:]))) if len(lv) > 2 else 0.2
        return vol, float(lv[-1] - lv.max())

    return ctx


def load_rows(path, ctx):
    rows = []
    for line in open(path):
        r = json.loads(line)
        r["lpred"] = np.log(np.maximum(r["pred"], FLOOR))
        r["lact"] = np.log(np.maximum(r["actual"], FLOOR))
        r["lq"] = {float(k): np.log(np.maximum(v, FLOOR)) for k, v in r.get("q", {}).items()}
        r["tier"] = bucket(r["rank"])
        r["vol"], r["below_peak"] = ctx(r["key"], r["origin"])
        rows.append(r)
    return rows


def alphas_for(level):
    return round((1 - level) / 2, 6), round(1 - (1 - level) / 2, 6)


def conformal_correction(residuals, level):
    """Finite-sample conformal quantile of a set of nonconformity scores."""
    n = len(residuals)
    k = min(int(np.ceil((n + 1) * level)), n)
    return float(np.sort(residuals)[k - 1])


class Bands:
    """A construction that turns calibration rows into a (lo, hi) band per row."""

    def __init__(self, name, key_fn, kind, min_group=60):
        self.name, self.key_fn, self.kind, self.min_group = name, key_fn, kind, min_group

    def fit(self, cal, level):
        groups = defaultdict(list)
        for r in cal:
            groups[self.key_fn(r)].append(r)
        self.table = {
            g: self._fit_group(v, level) for g, v in groups.items() if len(v) >= self.min_group
        }
        self.fallback = self._fit_group(cal, level)
        return self

    def _fit_group(self, sel, level):
        lo_a, hi_a = alphas_for(level)
        if self.kind == "resid":
            # log(actual) - log(point forecast), per horizon.
            R = np.vstack([r["lact"] - r["lpred"] for r in sel])
            return np.quantile(R, lo_a, axis=0), np.quantile(R, hi_a, axis=0)
        if self.kind == "cqr_asym":
            # The two sides get their own correction. A symmetric one cannot
            # repair a band that is right on the downside and too low on the
            # upside, which is exactly what the fitted quantiles do in ranks
            # 101-1000. Each tail is calibrated at 1 - a/2, so the union still
            # covers at 1 - a.
            half = 1 - (1 - level) / 2
            lo_adj, hi_adj = [], []
            for h in range(H):
                lo_adj.append(
                    conformal_correction(
                        np.array([r["lq"][lo_a][h] - r["lact"][h] for r in sel]), half
                    )
                )
                hi_adj.append(
                    conformal_correction(
                        np.array([r["lact"][h] - r["lq"][hi_a][h] for r in sel]), half
                    )
                )
            return np.array(lo_adj), np.array(hi_adj)
        # CQR: one nonconformity score per horizon, how far outside the
        # fitted band the truth fell (negative when comfortably inside).
        out = []
        for h in range(H):
            e = [max(r["lq"][lo_a][h] - r["lact"][h], r["lact"][h] - r["lq"][hi_a][h]) for r in sel]
            out.append(conformal_correction(np.array(e), level))
        return np.array(out)

    def apply(self, r, level):
        lo_a, hi_a = alphas_for(level)
        adj = self.table.get(self.key_fn(r), self.fallback)
        if self.kind == "resid":
            lo_q, hi_q = adj
            return r["lpred"] + lo_q, r["lpred"] + hi_q
        if self.kind == "cqr_asym":
            lo_adj, hi_adj = adj
            return r["lq"][lo_a] - lo_adj, r["lq"][hi_a] + hi_adj
        return r["lq"][lo_a] - adj, r["lq"][hi_a] + adj


class Direct(Bands):
    """The quantile boosters as they come, with no calibration at all."""

    def __init__(self):
        super().__init__("direct", lambda r: "*", "direct")

    def fit(self, cal, level):
        return self

    def apply(self, r, level):
        lo_a, hi_a = alphas_for(level)
        return r["lq"][lo_a], r["lq"][hi_a]


def vol_binner(cal):
    edges = np.quantile([r["vol"] for r in cal], [1 / 3, 2 / 3])
    return lambda r: (r["tier"], int(np.searchsorted(edges, r["vol"])))


def peak_bin(r):
    """Where the name sits relative to its own historic high, at the origin."""
    b = r["below_peak"]
    return "at peak" if b > -0.2 else ("off peak" if b > -0.7 else "long past")


def score(arm, test, level, groups_of=lambda r: (r["tier"], "ALL")):
    """Coverage, miss sides, width, interval score and asymmetry, by group."""
    a = 1 - level
    acc = defaultdict(lambda: defaultdict(list))
    for r in test:
        lo, hi = arm.apply(r, level)
        y = r["lact"]
        inside = (y >= lo) & (y <= hi)
        width = hi - lo
        pen = (2 / a) * (np.maximum(lo - y, 0) + np.maximum(y - hi, 0))
        upper = np.maximum(hi - r["lpred"], 0)
        for g in groups_of(r):
            acc[g]["cover"].append(inside)
            acc[g]["miss_lo"].append(y < lo)
            acc[g]["miss_hi"].append(y > hi)
            acc[g]["width"].append(width)
            acc[g]["score"].append(width + pen)
            acc[g]["upper"].append(upper)
            acc[g]["total"].append(np.maximum(width, 1e-12))
    out = {}
    for tier, d in acc.items():
        out[tier] = {
            "n": len(d["cover"]),
            "cover": float(np.mean(np.array(d["cover"]))),
            "miss_lo": float(np.mean(np.array(d["miss_lo"]))),
            "miss_hi": float(np.mean(np.array(d["miss_hi"]))),
            "width": float(np.mean(np.array(d["width"]))),
            "score": float(np.mean(np.array(d["score"]))),
            "asym": float(np.mean(np.array(d["upper"])) / np.mean(np.array(d["total"]))),
            "cover_h": np.array(d["cover"]).mean(axis=0),
            "width_h": np.array(d["width"]).mean(axis=0),
        }
    return out


def pinball(rows, alphas):
    """Mean pinball loss per alpha, in log space — the loss the boosters minimise."""
    out = {}
    for al in alphas:
        losses = []
        for r in rows:
            d = r["lact"] - r["lq"][al]
            losses.append(np.where(d >= 0, al * d, (al - 1) * d))
        out[al] = float(np.mean(np.array(losses)))
    return out


def point_skill(rows, key_fn):
    """poolSkill against naive, per tier — for comparing q50 with the L2 forecast."""
    acc = defaultdict(lambda: np.zeros(2))
    for r in rows:
        a = np.array(r["actual"], dtype=float)
        acc[r["tier"]] += (np.abs(key_fn(r) - a).sum(), np.abs(r["last"] - a).sum())
    return {t: 1 - acc[t][0] / acc[t][1] for t in TIERS if acc[t][1] > 0}


def crossing(rows, alphas):
    """Two ways a set of independently fitted quantiles can be incoherent."""
    al = sorted(alphas)
    cross = mono = n = 0
    for r in rows:
        Q = np.vstack([r["lq"][a] for a in al])
        cross += int((np.diff(Q, axis=0) < 0).any())
        widths = r["lq"][al[-1]] - r["lq"][al[0]]
        mono += int((np.diff(widths) < 0).any())
        n += 1
    return cross / n, mono / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--cal-origins", default="1995:2009")
    ap.add_argument("--test-origins", default="2010:2019")
    ap.add_argument("--levels", default="0.8,0.95")
    ap.add_argument("--by-horizon", action="store_true")
    ap.add_argument(
        "--conditional",
        action="store_true",
        help="repeat the table split by how far past its peak the name is",
    )
    ap.add_argument(
        "--compare",
        default=None,
        help="two arm names, e.g. 'direct,resid_tiervol' — bootstrap the gap between them",
    )
    a = ap.parse_args()

    cal_o = set(parse_origins(a.cal_origins))
    test_o = set(parse_origins(a.test_origins))
    levels = [float(x) for x in a.levels.split(",")]

    rows = load_rows(a.path, series_context())
    cal = [r for r in rows if r["origin"] in cal_o]
    test = [r for r in rows if r["origin"] in test_o]
    alphas = sorted(rows[0]["lq"]) if rows[0]["lq"] else []
    print(
        f"calibrate on {sorted(cal_o)[0]}-{sorted(cal_o)[-1]} (n={len(cal)}), "
        f"test on {sorted(test_o)[0]}-{sorted(test_o)[-1]} (n={len(test)}), alphas={alphas}"
    )

    if alphas:
        cross, mono = crossing(test, alphas)
        print(
            f"\nquantile coherence: {100 * cross:.2f}% of rows cross in alpha, "
            f"{100 * mono:.2f}% have a band that narrows as the horizon grows"
        )
        print("mean pinball loss (log space), fitted quantiles:")
        for al, v in pinball(test, alphas).items():
            print(f"  alpha={al:<6} {v:.4f}")
        if 0.5 in alphas:
            # A free side-effect worth reading: the pinball fit at alpha=0.5
            # is a conditional *median*, and poolSkill is an absolute-error
            # metric, so it is not obvious the squared-error fit should win.
            l2 = point_skill(test, lambda r: np.array(r["pred"]))
            q50 = point_skill(test, lambda r: np.array(r["q"]["0.5"]))
            print("\npoint forecast, poolSkill vs naive:")
            print(f"{'':14} " + " ".join(f"{t:>9}" for t in TIERS))
            for label, sc in (("l2 mean", l2), ("q50 median", q50)):
                print(f"{label:14} " + " ".join(f"{sc.get(t, float('nan')):9.3f}" for t in TIERS))

    binner = vol_binner(cal)

    def build_arms():
        arms = [
            Bands("resid_global", lambda r: "*", "resid"),
            Bands("resid_tiervol", binner, "resid"),
        ]
        if alphas:
            arms += [
                Direct(),
                Bands("cqr", lambda r: "*", "cqr"),
                Bands("cqr_tiervol", binner, "cqr"),
                Bands("cqr_asym", lambda r: "*", "cqr_asym"),
                Bands("cqr_asym_tier", lambda r: r["tier"], "cqr_asym"),
            ]
        return arms

    for level in levels:
        arms = build_arms()
        print(f"\n=== nominal {100 * level:.0f}% ===")
        print(
            f"{'arm':14} {'tier':9} {'n':>6} {'cover':>7} {'lo%':>6} {'hi%':>6} "
            f"{'width':>7} {'intScore':>9} {'asym':>6}"
        )
        for arm in arms:
            arm.fit(cal, level)
            sc = score(arm, test, level)
            for tier in TIERS + ("ALL",):
                if tier not in sc:
                    continue
                s = sc[tier]
                print(
                    f"{arm.name:14} {tier:9} {s['n']:6d} {s['cover']:7.3f} "
                    f"{100 * s['miss_lo']:6.1f} {100 * s['miss_hi']:6.1f} "
                    f"{s['width']:7.3f} {s['score']:9.3f} {s['asym']:6.3f}"
                )
            if a.by_horizon:
                s = sc["ALL"]
                print(
                    f"{'':14} {'  h1-h5':9} "
                    + " cover "
                    + " ".join(f"{v:.3f}" for v in s["cover_h"])
                    + "  width "
                    + " ".join(f"{v:.2f}" for v in s["width_h"])
                )
        if a.conditional:
            conditional_table(build_arms(), cal, test, level)
        if a.compare:
            compare(build_arms(), cal, test, level, a.compare.split(","))


def compare(arms, cal, test, level, pair, draws=5000, seed=0):
    """Is the interval-score gap between two arms bigger than the sample noise?

    Resamples *names*, not name-origins, for the reason `paired.py` gives: one
    name's overlapping five-year windows are not independent draws.
    """
    a_name, b_name = pair
    by = {arm.name: arm.fit(cal, level) for arm in arms}
    if a_name not in by or b_name not in by:
        return
    rng = np.random.default_rng(seed)
    print(
        f"\n=== nominal {100 * level:.0f}%: {a_name} vs {b_name}, cluster bootstrap over names ==="
    )
    print(f"{'tier':10} {'names':>6} {a_name:>14} {b_name:>14} {'diff [2.5%, 97.5%]':>30}")
    for tier in TIERS + ("ALL",):
        units = defaultdict(lambda: np.zeros(3))
        for r in test:
            if tier != "ALL" and r["tier"] != tier:
                continue
            row = []
            for name in (a_name, b_name):
                lo, hi = by[name].apply(r, level)
                y = r["lact"]
                pen = (2 / (1 - level)) * (np.maximum(lo - y, 0) + np.maximum(y - hi, 0))
                row.append(float((hi - lo + pen).sum()))
            units[r["key"]] += (row[0], row[1], len(r["lact"]))
        if not units:
            continue
        sa, sb, n = np.array(list(units.values())).T
        idx = rng.integers(0, len(sa), size=(draws, len(sa)))
        # Lower interval score is better, so the difference is b - a: positive
        # means the first arm is the better one.
        diff = sb[idx].sum(1) / n[idx].sum(1) - sa[idx].sum(1) / n[idx].sum(1)
        lo_ci, hi_ci = np.percentile(diff, [2.5, 97.5])
        print(
            f"{tier:10} {len(units):6d} {sa.sum() / n.sum():14.3f} {sb.sum() / n.sum():14.3f} "
            f"{diff.mean():+9.3f} [{lo_ci:+.3f}, {hi_ci:+.3f}]  P={100 * (diff > 0).mean():.0f}%"
        )


def conditional_table(arms, cal, test, level):
    """The test the marginal table cannot run: is the band right *per situation*?"""
    print(f"\n=== nominal {100 * level:.0f}%, split by position relative to the name's peak ===")
    print(
        f"{'arm':14} {'bin':10} {'n':>6} {'cover':>7} {'lo%':>6} {'hi%':>6} {'intScore':>9} {'asym':>6}"
    )
    for arm in arms:
        arm.fit(cal, level)
        sc = score(arm, test, level, groups_of=lambda r: (peak_bin(r),))
        for b in ("at peak", "off peak", "long past"):
            if b not in sc:
                continue
            s = sc[b]
            print(
                f"{arm.name:14} {b:10} {s['n']:6d} {s['cover']:7.3f} "
                f"{100 * s['miss_lo']:6.1f} {100 * s['miss_hi']:6.1f} "
                f"{s['score']:9.3f} {s['asym']:6.3f}"
            )


if __name__ == "__main__":
    main()
