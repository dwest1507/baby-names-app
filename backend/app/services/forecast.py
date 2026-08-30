"""ARIMA popularity forecasting, ported from the original Streamlit app.

Fits a small grid of ARIMA models selected by AICc, forecasts 5 years ahead
with 80%/95% confidence intervals, and validates on a 5-year holdout. Results
are cached per (name, sex) since fitting is CPU-bound.
"""

import logging
import warnings
from functools import lru_cache

import numpy as np
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss

from . import queries

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

FORECAST_YEARS = 5
VALIDATION_YEARS = 5
MIN_HISTORY_YEARS = 10
MAX_P = 3
MAX_D = 2
MAX_Q = 3


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

    adf_pvalue = adfuller(series, autolag="AIC")[1]
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
    """Grid search over (p, d, q) around the optimal differencing, selected by AICc."""
    optimal_d = _find_optimal_differencing(series)
    d_range = range(max(0, optimal_d - 1), min(MAX_D, optimal_d + 1) + 1)

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
    """Refit on all but the last VALIDATION_YEARS and score against the holdout."""
    if len(values) < MIN_HISTORY_YEARS + VALIDATION_YEARS:
        return None

    train, test = values[:-VALIDATION_YEARS], values[-VALIDATION_YEARS:]
    processed, log_applied = _preprocess(train)
    model, _ = _fit_best_model(processed)
    if model is None:
        return None

    try:
        predicted = _forecast(model, log_applied, VALIDATION_YEARS)["mean"]
    except Exception:
        return None

    errors = test - predicted
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    mape = float(np.mean(np.abs(errors / np.maximum(test, 1e-12))) * 100)

    test_years = years[-VALIDATION_YEARS:]
    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "points": [
            {"year": int(y), "actual": float(a), "predicted": float(p)}
            for y, a, p in zip(test_years, test, predicted, strict=True)
        ],
    }


@lru_cache(maxsize=256)
def forecast_name(name: str, sex: str) -> dict | None:
    """Build the full forecast payload for a name, or None when it has no history."""
    history = queries.get_name_history(name, sex)
    if not history:
        return None

    years = [row["year"] for row in history]
    values = np.array([row["popularity_percent"] for row in history], dtype=float)

    payload: dict = {
        "name": history[0]["name"],
        "sex": sex,
        "history": [
            {"year": int(y), "value": float(v)} for y, v in zip(years, values, strict=True)
        ],
        "forecast": [],
        "validation": None,
        "model": None,
    }

    if len(values) < MIN_HISTORY_YEARS:
        return payload

    processed, log_applied = _preprocess(values)
    model, params = _fit_best_model(processed)
    if model is None:
        return payload

    try:
        forecast = _forecast(model, log_applied, FORECAST_YEARS)
    except Exception as e:
        logger.warning("ARIMA forecasting failed for %s (%s): %s", name, sex, e)
        return payload

    last_year = years[-1]
    future_years = range(last_year + 1, last_year + FORECAST_YEARS + 1)
    ci80, ci95 = forecast["intervals"][0.8], forecast["intervals"][0.95]
    payload["forecast"] = [
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
    payload["model"] = {
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
    payload["validation"] = _validate(values, years)
    return payload
