"""The two halves of forecasting that the request path actually uses.

Forecasts are fitted offline by the precompute batch and stored, so serving
one is a lookup: read history fresh, read the stored blob, compose. Nothing
here fits anything, and nothing here imports a fitting library — which is why
no fitting library is a runtime dependency at all. The pooled model that
produces the stored blob lives in
`scripts/forecast/pooled.py`, outside the application package and outside the
container image. See docs/adr/0004-forecasts-as-a-build-artifact.md and
docs/adr/0010-a-pooled-model-replaces-per-name-arima.md.
"""

# The batch imports this to decide how much history a holdout needs, so the
# eligibility rule has one definition rather than two that can drift.
MIN_HISTORY_YEARS = 10


def is_eligible(years: list[int], latest_year: int | None) -> bool:
    """Whether a name/sex's observed years qualify it for a forecast.

    A forecast is produced only for a name observed in the newest year present
    in the data, with at least `MIN_HISTORY_YEARS` observed years. This also
    guarantees no forecast can land on a year that has already occurred, since
    every eligible name's last observation is the newest year. See
    docs/adr/0001-forecast-only-names-in-current-use.md.
    """
    return bool(years) and years[-1] == latest_year and len(years) >= MIN_HISTORY_YEARS


def build_response(
    sex: str,
    history: list[dict],
    stored: dict | None,
    calibration: dict | None = None,
    model_card: dict | None = None,
) -> dict:
    """Compose the API response from history read fresh plus a stored blob.

    `stored` is the JSON-decoded `forecasts.payload` for this name/sex, or
    None when there is no row — either because the name was ineligible when
    the batch ran, or because it has no forecast for any other reason. Either
    way the response shape matches what the endpoint always returned: an
    empty forecast list rather than a missing key. No fitting happens here.

    `calibration` is the measured interval coverage for *this name's*
    stratum — its popularity tier and volatility bin
    (`queries.get_calibration`) — rather than one population figure shared by
    every name. It is None only when there is no forecast to draw bands for.
    See docs/adr/0011-conformal-bands-keyed-by-strata.md: the frontend must
    label the shaded bands with this measured coverage, not the nominal
    80%/95%, and the figure it labels them with is the one measured for names
    like this one.

    `stored["stratum"]` is the name's *own* calibration stratum — its
    popularity tier at the origin and its volatility bin — which is not the
    same fact as the stratum `calibration` describes. A name whose own cell
    was too thin to earn a band of its own is served the whole population's,
    and the calibration row then names `*`. The page reports both: the tier
    and bin the name is in, beside the coverage measured for the band it was
    actually given. `(None, None)` on an artifact published before the
    columns existed (ADR 0006), which costs the label and nothing else.

    `model_card` is what the batch can honestly say about the model itself
    (`queries.get_model_card`). One pooled model forecasts every name, so it
    describes the batch rather than this name, and it is served under `model`
    where a per-name ARIMA fit's order and residual diagnostics used to go.

    `track_record` is what the model said about each year, keyed by horizon:
    for horizon *h*, year *Y*'s entry is the prediction made at origin
    *Y − h*. The batch stores it as a start year and parallel arrays; this is
    where it becomes one object per year — the one place the stored payload is
    not served verbatim, and composition rather than fitting, so ADR 0004
    holds. A year the model was never checked on has no entry. See
    docs/adr/0012-a-track-record-replaces-the-holdout-on-the-page.md.

    Every entry, and every forecast point, carries a `projected_rank` beside
    its share: the rank that projection earned against the whole field
    observed at its origin. Ranking is a statement about every name at once,
    so it happens in the batch and nothing is ranked here. See
    docs/adr/0013-projected-rank-against-a-frozen-field.md.
    """
    return {
        "name": history[0]["name"],
        "sex": sex,
        "history": [
            {"year": int(row["year"]), "value": float(row["popularity_percent"])} for row in history
        ],
        "forecast": stored["forecast"] if stored else [],
        "validation": stored["validation"] if stored else None,
        "model": model_card if stored else None,
        "calibration": calibration if stored else None,
        "stratum": _stratum(stored) if stored else None,
        "track_record": _track_record(stored) if stored else {},
    }


def _track_record(stored: dict) -> dict[str, list[dict]]:
    return {
        horizon: [
            {
                "year": series["start"] + offset,
                "projected_share": share,
                "projected_rank": rank,
            }
            for offset, (share, rank) in enumerate(
                zip(series["projected_share"], series["projected_rank"], strict=True)
            )
            if share is not None
        ]
        for horizon, series in stored.get("track_record", {}).items()
    }


def _stratum(stored: dict) -> dict | None:
    tier, volatility_bin = stored.get("stratum", (None, None))
    if tier is None or volatility_bin is None:
        return None
    return {"tier": tier, "volatility_bin": int(volatility_bin)}
