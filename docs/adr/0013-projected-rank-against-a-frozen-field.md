# 13. Projected rank is ranked against a frozen field, not against the forecastable names

Date: 2026-09-12

## Status

Proposed

Builds on [1. Forecast only names in current use](0001-forecast-only-names-in-current-use.md)
and [12. A track record, indexed by horizon, replaces the holdout on the search page](0012-a-track-record-replaces-the-holdout-on-the-page.md).

## Context

"Will it still be in the top ten?" is the question visitors actually have, and the site could not
answer it. The model predicts a name's **share of births**; `popularity_rank` is a recorded column
on `names`. Nothing forecasts a rank, and nothing can derive one from a single name's forecast — a
rank is a statement about a name's position among all the others, so it can only be computed across
every name at once, in the batch.

The tempting way to compute it is to rank the eligible names against each other: they are the names
with forecasts, they are already in hand at every origin, and it costs one sort. That is wrong in a
way that is invisible where it is usually looked at and severe everywhere else. ADR 0001 measured
the eligible set: **24,721 of 116,550 name/sex pairs**, about one in five. But eligibility is not
uniform — **1,996 of the 2,000 top-1000-ranked pairs of 2024 remain eligible**, 99.8%. So ranking
against the eligible set alone is essentially exact at the head of the distribution and increasingly
generous below it, because each un-forecastable competitor that is missing from the field lifts
every name beneath it by one place.

ADR 0012 is what makes this intolerable rather than merely imprecise. The track record puts a
**projected rank beside an actual rank in adjacent columns**, row after row. A systematic upward
bias in one of them does not read as a ranking artifact. It reads as *the model is always too
optimistic* — an accusation of the forecast, caused entirely by who was allowed into the race.

## Decision

**A projected rank is the rank its projected share earns against every name observed at the origin,
with names that cannot be forecast held at their last observed share.**

The field at origin `O` for horizon `h` is the union of:

- every name eligible at `O`, entered at its **projected share** for `O + h`; and
- every other name observed at `O`, entered at its **share as observed at `O`**.

Freezing is the assumption that says what it does not know. A name the model cannot forecast is not
thereby a name that vanishes, and entering it at its last observed share asserts only that it
carries on as it was — which is the same naive baseline the whole model is scored against. The
alternative, dropping it, asserts that it disappears, which is a much stronger claim and always
wrong in the same direction.

The field is built once per `(origin, horizon, sex)` in the batch, where the full series are already
in memory from `pooled.stream_series`.

## Considered options

**Rank against eligible names only.** One sort per slice and no extra data. Rejected: unbiased in
the top 1000 and progressively optimistic below it, which ADR 0012's side-by-side columns convert
into an apparent indictment of the model.

**Publish a projected rank only inside the top 1000, where the naive version is accurate.** Honest
and cheap, and it blanks the column for most of the database — including every name whose owner is
most curious about whether it is climbing.

**Drop projected rank.** Rejected: it removes the most intuitive question the page can answer, to
avoid a bias that is fixable with one sorted array per slice.

## Consequences

- **A residual bias remains and cannot be removed.** Names that first appear between `O` and `O + h`
  are absent from the projected field entirely, because a name that does not yet exist cannot be
  forecast or frozen. Real arrivals push incumbents down; projected arrivals cannot. The effect is
  small at the head, where debuts are rare, and it is in the same direction as the bias this ADR
  removes — it is smaller, not absent.
- **Projected rank is only as good as the frozen assumption for the tail.** Ranks deep in the field
  are built largely from frozen shares, so they move less than reality does. The figure is honest
  about position, not about churn.
- **Ranking is a batch-only operation, and stays one.** Ranks are stored per track record entry and
  per forecast point; the request path remains a lookup and ADR 0004 holds.
- **`model_evaluation` and the deploy gate are unaffected.** Skill is measured on shares, not ranks,
  so `verify_db.py` has nothing new to certify.

## Related

- Parent PRD: dwest1507/baby-names-app#59
- Depends on: ADR 0001 (the eligibility rule and its measured selectivity), ADR 0012 (the track
  record that puts projected and actual ranks side by side), ADR 0004 (batch-only computation)
- Introduces: CONTEXT.md "Projected Rank"
