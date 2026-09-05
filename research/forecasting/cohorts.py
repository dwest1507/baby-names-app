"""Cohort aggregates: how the *neighbourhood* of a name is moving.

Name popularity moves in correlated clusters — endings (-ayden, -ella), initial
letters, and the sex's overall share of births all trend together, and a
per-series model cannot see any of it. This builds, for every year, the total
share held by each cohort, so a forecaster can ask "is this name rising because
*it* is rising, or because everything that sounds like it is?"

Aggregates are built from the extracted series (names in current use with at
least 15 observed years), not from the full table: that is where essentially all
of the share mass sits, and it keeps the benchmark reliant on one extract.
"""

import os

import numpy as np

from data import SERIES, WORK, load_all

YEARS = np.arange(1880, 2025)
Y0 = YEARS[0]
OUT = os.path.join(WORK, "cohorts.npz")


def cohort_keys(name, sex):
    """The cohorts a name belongs to: ending, initial, and its sex."""
    return (f"e2:{sex}:{name[-2:]}", f"i1:{sex}:{name[0]}", f"sex:{sex}")


def build():
    series = load_all()
    totals = {}
    own = {}
    for key, years, vals, _rank in series:
        name, sex = str(key).split("|")
        idx = years - Y0
        own[str(key)] = (idx, vals)
        for ck in cohort_keys(name, sex):
            arr = totals.setdefault(ck, np.zeros(len(YEARS)))
            arr[idx] += vals
    names = sorted(totals)
    matrix = np.vstack([totals[n] for n in names])
    np.savez_compressed(OUT, cohorts=np.array(names), totals=matrix, years=YEARS)
    print(f"{len(names)} cohorts x {len(YEARS)} years -> {OUT}")


class Cohorts:
    """Leave-one-out cohort shares, so a name never predicts itself."""

    def __init__(self):
        d = np.load(OUT, allow_pickle=True)
        self.index = {str(n): i for i, n in enumerate(d["cohorts"])}
        self.totals = d["totals"]

    def series(self, key, upto_year):
        """Per-cohort share history for `key`, with the name's own share removed."""
        name, sex = str(key).split("|")
        n = upto_year - Y0 + 1
        out = []
        for ck in cohort_keys(name, sex):
            i = self.index.get(ck)
            out.append(self.totals[i, :n].copy() if i is not None else np.zeros(n))
        return out


if __name__ == "__main__":
    if not os.path.exists(SERIES):
        raise SystemExit("run extract_series.py first")
    build()
