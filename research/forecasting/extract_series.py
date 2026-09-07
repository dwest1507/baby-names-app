"""Extract per-name series from the real DB into a compact npz for backtesting."""

import os
import sqlite3
from collections import defaultdict

import numpy as np

from data import DB, WORK
from data import SERIES as OUT

os.makedirs(WORK, exist_ok=True)

c = sqlite3.connect(DB)
c.row_factory = sqlite3.Row
rank2024 = {}
for r in c.execute(
    "select lower(name) n, sex, popularity_rank, total_count from names where year=2024"
):
    rank2024[(r["n"], r["sex"])] = (r["popularity_rank"], r["total_count"])

groups = defaultdict(list)
for r in c.execute(
    "select lower(name) n, sex, year, popularity_percent from names order by name, sex, year"
):
    groups[(r["n"], r["sex"])].append((r["year"], r["popularity_percent"]))

keys, years_l, vals_l, ranks, counts = [], [], [], [], []
for k, rows in groups.items():
    if k not in rank2024:  # must be in current use (ADR 0001 eligibility)
        continue
    if len(rows) < 15:
        continue
    keys.append(k)
    years_l.append(np.array([x[0] for x in rows], dtype=np.int32))
    vals_l.append(np.array([x[1] for x in rows], dtype=np.float64))
    ranks.append(rank2024[k][0])
    counts.append(rank2024[k][1])

print("series:", len(keys))
np.savez_compressed(
    OUT,
    keys=np.array([f"{n}|{s}" for n, s in keys]),
    years=np.array(years_l, dtype=object),
    vals=np.array(vals_l, dtype=object),
    rank=np.array(ranks),
    count=np.array(counts),
    allow_pickle=True,
)
print("wrote", OUT)
