"""The ARIMA popularity pipeline, ported from the original Streamlit app.

Fits a small grid of ARIMA models selected by AICc, forecasts 5 years ahead
with 80%/95% confidence intervals, and validates on a 5-year holdout.

This is the offline half of forecasting and the only place fitting happens.
It runs from `scripts/precompute_forecasts.py`, is never imported by the
application package, and is not copied into the production image, so
`statsmodels` and `scipy` are build/dev dependencies rather than runtime
ones. The request path composes its response from the stored result via
`app.services.forecast.build_response`. See
docs/adr/0004-forecasts-as-a-build-artifact.md.

`research/forecasting/methods.py` imports this module directly for its
`current` baseline arm, so that the harness benchmarks the shipped pipeline
rather than a copy of it.
"""

import logging
import warnings

import numpy as np
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tools.sm_exceptions import ModelWarning
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss

from app.services.forecast import MIN_HISTORY_YEARS

# Grid-searching ARIMA orders makes statsmodels complain constantly about
# non-convergence, non-invertible starting parameters and the like; those are
# expected here since poor fits are simply discarded by the AICc search. Keep
# the suppression scoped to statsmodels so warnings from the rest of the app
# still surface.
warnings.filterwarnings("ignore", category=ModelWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module=r"statsmodels\..*")

logger = logging.getLogger(__name__)

FORECAST_YEARS = 5
VALIDATION_YEARS = 5
MAX_P = 2
MAX_D = 2
MAX_Q = 2


def _preprocess(series: np.ndarray) -> tuple[np.ndarray, bool]:
    """Log-transform series with high relative variance (multiplicative trends)."""
    if (series > 0).all() and series.std() / series.mean() > 0.5:
        return np.log1p(series), True
    return series, False


def _inverse(series: np.ndarray, log_applied: bool) -> np.ndarray:
    return np.expm1(series) if log_applied else series


def _check_stationarity(series: np.ndarray) -> tuple[bool, float, float]:
    if len(series) < 4:
        return False, 1.0, 1.0

    # Both tests reject degenerate input (a constant series raises outright), and
    # differencing in _find_optimal_differencing can easily produce one.
    try:
        adf_pvalue = adfuller(series, autolag="AIC")[1]
    except Exception:
        adf_pvalue = 1.0
    try:
        kpss_pvalue = kpss(series, regression="c", nlags="auto")[1]
    except Exception:
        kpss_pvalue = 1.0

    is_stationary = bool(adf_pvalue < 0.05 and kpss_pvalue > 0.05)
    return is_stationary, float(adf_pvalue), float(kpss_pvalue)


def _find_optimal_differencing(series: np.ndarray, max_d: int = MAX_D) -> int:
    current = series.copy()
    for d in range(max_d + 1):
        is_stationary, _, _ = _check_stationarity(current)
        if is_stationary:
            return d
        current = np.diff(current)
    return max_d


def _fit_best_model(series: np.ndarray):
    """Grid search over (p, q) at the tested differencing order, selected by AICc."""
    # Only the differencing order the ADF/KPSS tests actually chose. The grid
    # used to sweep optimal_d +/- 1 and pick the winner by AICc, but AICc is
    # not comparable across different `d`: differencing changes the data the
    # likelihood is computed on, so those three sets of models were never on a
    # common scale. Searching the one tested value is both correct and three
    # times cheaper. See docs/adr/0007-precompute-batch-runs-in-parallel.md.
    d_range = (_find_optimal_differencing(series),)

    best_aicc = float("inf")
    best_params = None
    best_model = None

    n = len(series)
    for p in range(MAX_P + 1):
        for d in d_range:
            for q in range(MAX_Q + 1):
                try:
                    fitted = ARIMA(series, order=(p, d, q)).fit()
                    k = p + q + 1
                    if n - k - 1 <= 0:
                        continue
                    aicc = fitted.aic + (2 * k * (k + 1)) / (n - k - 1)
                    if aicc < best_aicc:
                        best_aicc = aicc
                        best_params = (p, d, q)
                        best_model = fitted
                except Exception:
                    continue

    if best_model is None:
        try:
            best_params = (1, 1, 1)
            best_model = ARIMA(series, order=best_params).fit()
        except Exception:
            return None, None
    return best_model, best_params


def _residual_diagnostics(model) -> dict:
    residuals = model.resid
    diagnostics = {}

    try:
        lb = acorr_ljungbox(residuals, lags=min(10, len(residuals) // 4))
        p_value = float(lb["lb_pvalue"].iloc[-1])
        diagnostics["ljung_box"] = {"p_value": p_value, "is_white_noise": bool(p_value > 0.05)}
    except Exception:
        diagnostics["ljung_box"] = {"p_value": 1.0, "is_white_noise": True}

    try:
        _, jb_pvalue = stats.jarque_bera(residuals)
        diagnostics["normality"] = {
            "p_value": float(jb_pvalue),
            "is_normal": bool(jb_pvalue > 0.05),
        }
    except Exception:
        diagnostics["normality"] = {"p_value": 1.0, "is_normal": True}

    try:
        arch = het_arch(residuals)
        diagnostics["heteroscedasticity"] = {
            "p_value": float(arch[1]),
            "is_homoscedastic": bool(arch[1] > 0.05),
        }
    except Exception:
        diagnostics["heteroscedasticity"] = {"p_value": 1.0, "is_homoscedastic": True}

    diagnostics["overall_quality"] = (
        diagnostics["ljung_box"]["is_white_noise"]
        and diagnostics["normality"]["is_normal"]
        and diagnostics["heteroscedasticity"]["is_homoscedastic"]
    )
    return diagnostics


def _forecast(model, log_applied: bool, steps: int) -> dict:
    result = model.get_forecast(steps=steps)
    mean = _inverse(np.asarray(result.predicted_mean), log_applied)

    intervals = {}
    for level in (0.8, 0.95):
        conf = np.asarray(result.conf_int(alpha=1 - level))
        intervals[level] = {
            "lower": _inverse(conf[:, 0], log_applied),
            "upper": _inverse(conf[:, 1], log_applied),
        }
    return {"mean": mean, "intervals": intervals}


def _validate(values: np.ndarray, years: list[int]) -> dict | None:
    """Refit on all but the last VALIDATION_YEARS and score against the holdout.

    This is also the backtest that calibrates the published intervals (see
    docs/adr/0005-truthful-confidence-intervals.md): the same training-only fit
    used to score MAE/RMSE/MAPE also produces the 80%/95% intervals it *would*
    have published, so `coverage` records whether each holdout point actually
    fell inside them. `scripts/precompute_forecasts.py` aggregates `coverage`
    across every eligible name into the `calibration` table and strips it
    before storing this dict, so it never reaches the API response.

    `skill` compares the model's holdout MAE against a naive/persistence
    baseline — the last training-observed value repeated for every holdout
    year, the standard "no change" forecast. `skill = 1 - model_mae /
    naive_mae`: 0 means the model does no better than assuming nothing
    changes, negative means it does worse.
    """
    if len(values) < MIN_HISTORY_YEARS + VALIDATION_YEARS:
        return None

    train, test = values[:-VALIDATION_YEARS], values[-VALIDATION_YEARS:]
    processed, log_applied = _preprocess(train)
    model, _ = _fit_best_model(processed)
    if model is None:
        return None

    try:
        holdout_forecast = _forecast(model, log_applied, VALIDATION_YEARS)
    except Exception:
        return None

    predicted = holdout_forecast["mean"]
    errors = test - predicted
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    mape = float(np.mean(np.abs(errors / np.maximum(test, 1e-12))) * 100)

    naive_predicted = np.full(VALIDATION_YEARS, train[-1])
    naive_mae = float(np.mean(np.abs(test - naive_predicted)))
    skill = float(1 - mae / naive_mae) if naive_mae > 0 else 0.0

    coverage = {}
    for level in (0.8, 0.95):
        interval = holdout_forecast["intervals"][level]
        coverage[str(level)] = [
            bool(lo <= actual <= hi)
            for lo, hi, actual in zip(interval["lower"], interval["upper"], test, strict=True)
        ]

    test_years = years[-VALIDATION_YEARS:]
    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "skill": skill,
        "points": [
            {"year": int(y), "actual": float(a), "predicted": float(p)}
            for y, a, p in zip(test_years, test, predicted, strict=True)
        ],
        "coverage": coverage,
    }


def fit_forecast(history: list[dict]) -> dict:
    """Fit an ARIMA model and produce the forecast/validation/model blob.

    This is the CPU-bound half of forecasting — the part that must run only
    once, offline, from `scripts/precompute_forecasts.py`, rather than on the
    request path. Callers are responsible for checking `is_eligible` first;
    this function fits unconditionally on whatever history it is given, and
    the result is exactly what is stored in the `forecasts` table (history
    itself excluded — the caller already has it).
    """
    years = [row["year"] for row in history]
    values = np.array([row["popularity_percent"] for row in history], dtype=float)

    result: dict = {"forecast": [], "validation": None, "model": None}

    processed, log_applied = _preprocess(values)
    model, params = _fit_best_model(processed)
    if model is None:
        return result

    try:
        forecast = _forecast(model, log_applied, FORECAST_YEARS)
    except Exception:
        logger.warning("ARIMA forecasting failed", exc_info=True)
        return result

    last_year = years[-1]
    future_years = range(last_year + 1, last_year + FORECAST_YEARS + 1)
    ci80, ci95 = forecast["intervals"][0.8], forecast["intervals"][0.95]
    result["forecast"] = [
        {
            "year": int(year),
            "mean": float(max(forecast["mean"][i], 0.0)),
            "lo80": float(max(ci80["lower"][i], 0.0)),
            "hi80": float(max(ci80["upper"][i], 0.0)),
            "lo95": float(max(ci95["lower"][i], 0.0)),
            "hi95": float(max(ci95["upper"][i], 0.0)),
        }
        for i, year in enumerate(future_years)
    ]

    is_stationary, adf_p, kpss_p = _check_stationarity(processed)
    result["model"] = {
        "order": list(params),
        "aic": float(model.aic),
        "bic": float(model.bic),
        "log_applied": log_applied,
        "diagnostics": _residual_diagnostics(model),
        "stationarity": {
            "is_stationary": is_stationary,
            "adf_pvalue": adf_p,
            "kpss_pvalue": kpss_p,
        },
    }
    result["validation"] = _validate(values, years)
    return result
