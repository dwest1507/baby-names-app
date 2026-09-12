"""The two halves of forecasting that the request path actually uses.

Forecasts are fitted offline by the precompute batch and stored, so serving
one is a lookup: read history fresh, read the stored blob, compose. Nothing
here fits anything, and nothing here imports a fitting library — which is why
`statsmodels` and `scipy` are not runtime dependencies at all. The ARIMA
pipeline that produces the stored blob lives in `scripts/forecast/arima.py`,
outside the application package and outside the container image. See
docs/adr/0004-forecasts-as-a-build-artifact.md.
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
    sex: str, history: list[dict], stored: dict | None, calibration: dict | None = None
) -> dict:
    """Compose the API response from history read fresh plus a stored blob.

    `stored` is the JSON-decoded `forecasts.payload` for this name/sex, or
    None when there is no row — either because the name was ineligible when
    the batch ran, or because it has no forecast for any other reason. Either
    way the response shape matches what the endpoint always returned: an
    empty forecast list rather than a missing key. No fitting happens here.

    `calibration` is the batch's measured interval coverage
    (`queries.get_calibration`), the same for every name — it is None only
    when there is no forecast to draw bands for. See
    docs/adr/0005-truthful-confidence-intervals.md: the frontend must label
    the shaded bands with this measured coverage, not the nominal 80%/95%.
    """
    return {
        "name": history[0]["name"],
        "sex": sex,
        "history": [
            {"year": int(row["year"]), "value": float(row["popularity_percent"])} for row in history
        ],
        "forecast": stored["forecast"] if stored else [],
        "validation": stored["validation"] if stored else None,
        "model": stored["model"] if stored else None,
        "calibration": calibration if stored else None,
    }
