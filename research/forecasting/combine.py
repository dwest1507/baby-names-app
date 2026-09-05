"""Log-space combinations of forecasts already computed by other runs."""

import json
import sys

import numpy as np

FLOOR = 1e-12


def main():
    paths = sys.argv[1:-1]
    out = sys.argv[-1]
    rows = [json.loads(line) for p in paths for line in open(p)]
    by = {}
    for r in rows:
        by.setdefault((r["key"], r["origin"]), {})[r["method"]] = r
    COMBOS = {
        "combo_pooled_ens": ("pooled_ridge", "ens_all4"),
        "combo_pooled_current": ("pooled_ridge", "current"),
        "combo_pooled_dampfam": ("pooled_ridge", "ens_dampfam"),
        "combo_pooled_ens_naive": ("pooled_ridge", "ens_all4", "naive"),
        "combo_all": ("pooled_ridge", "ens_all4", "current", "ets_log_phi80"),
        "combo_t1k_ens": ("pooled_t1k", "ens_all4"),
        "combo_t1k_pooled_ens": ("pooled_t1k", "pooled_ridge", "ens_all4"),
        "combo_t1k_current": ("pooled_t1k", "current"),
    }
    n = 0
    with open(out, "w") as fh:
        for d in by.values():
            for name, parts in COMBOS.items():
                if not all(p in d for p in parts):
                    continue
                L = np.mean([np.log(np.maximum(d[p]["pred"], FLOOR)) for p in parts], axis=0)
                base = d[parts[0]]
                fh.write(
                    json.dumps(
                        {
                            **{
                                kk: base[kk]
                                for kk in ("key", "rank", "origin", "actual", "last", "years")
                            },
                            "method": name,
                            "secs": 0.0,
                            "pred": np.exp(L).tolist(),
                        }
                    )
                    + "\n"
                )
                n += 1
            # shrunk pooled
            if "pooled_ridge" in d:
                base = d["pooled_ridge"]
                for w in (0.7,):
                    L = w * np.log(np.maximum(base["pred"], FLOOR)) + (1 - w) * np.log(
                        max(base["last"], FLOOR)
                    )
                    fh.write(
                        json.dumps(
                            {
                                **{
                                    kk: base[kk]
                                    for kk in ("key", "rank", "origin", "actual", "last", "years")
                                },
                                "method": f"pooled_s{int(w * 100)}",
                                "secs": 0.0,
                                "pred": np.exp(L).tolist(),
                            }
                        )
                        + "\n"
                    )
                    n += 1
    print("wrote", n, "rows ->", out)


if __name__ == "__main__":
    main()
