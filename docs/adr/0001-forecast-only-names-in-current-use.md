# 1. Forecast only names in current use

Date: 2026-08-30

## Status

Accepted

## Context

The names table is padded. The data pipeline cross-joins every name/sex pair with every year
from 1880 to 2024 and zero-fills the gaps, producing 14,750,273 fabricated rows against
2,149,477 real observations. The app treated those invented rows as observations.

Two defects followed from that, and both are visible to a visitor:

- **Forecasts were fitted on fiction.** A padded series is dominated by decades of zeros, so
  the AICc grid search selected mean-reverting models that pull the forecast back toward an
  inflated long-run average. The app predicted *Karen* rising fivefold over five years beside a
  holdout MAPE of 284%, drawn as a confident dashed line. The band labelled 95% covered 71% of
  actual values across 300 interval checks.
- **Every name looked like it was still in use.** Because the padding runs to the final year,
  every name's last row was the final year, so every name got a forecast — including names last
  actually recorded decades ago.

Removing the padding at the query layer fixes the first defect but exposes a second one: a name
last recorded in 1993 would then be forecast for 1994–1998, years that have already happened.

A rule is therefore needed for *which* names are forecast at all.

## Decision

**A forecast is produced only for a name observed in the newest year present in the data, that
also has at least `MIN_HISTORY_YEARS` (10) observed years.**

- The newest year is **read from the data** (`queries.get_latest_data_year()`, the maximum year
  with a recorded count), never hardcoded. Next year's data refresh needs no code change.
- A name that fails the rule returns its history with an **empty forecast list**. The payload
  shape is unchanged; the frontend explains which of the two reasons applies.
- The forecast service **does not reconstruct or re-pad the series** to a full year range. An
  earlier position in the design session was to re-pad inside the forecast service so that
  output stayed byte-identical to the old behaviour. That was **reversed**: the padding is
  itself the defect. Removing it raises measured interval coverage from 0.710 to 0.767, and it
  un-blocks the log transform, whose `(series > 0).all()` gate was denying a multiplicative
  model to 694 of the top 1000 names of 2024 — precisely the modern, fast-growing names where
  it matters most.

## Consequences

- **24,721 name/sex pairs are eligible** for a forecast, out of 116,550 pairs in the data.
- **Every eligible name's last observation is the newest year**, so the five-year horizon is
  uniform across the site and no forecast can land on a year that has already occurred.
- **What is excluded is negligible.** The largest name excluded for recency had 34 births in its
  final year, and 1,996 of the 2,000 top-1000-ranked pairs of 2024 remain eligible.
- **The minimum-history guard becomes live code for the first time.** Under padding every series
  was 145 points long, so the guard could never fire.
- Two absences of a forecast must be told apart in the UI — "not in current use" and "not enough
  history" — because they are different facts about the name, and a silent gap reads as a bug.
- Forecast intervals are still labelled with their nominal level. Measured coverage improves but
  is not yet 95%; labelling intervals with their true coverage is tracked separately.
- The rule is defined against the newest year *in the data*, not the current calendar year. Data
  releases lag by about a year, so a forecast horizon may begin in a year that has already
  started. Anchoring to the data is deliberate: it keeps the horizon uniform across names and
  keeps eligibility reproducible against a fixed database artifact.

## Related

- Parent PRD: dwest1507/baby-names-app#5
- Implemented by: dwest1507/baby-names-app#6
