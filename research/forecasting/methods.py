"""Candidate forecasting methods. Each takes train values (1-D array, oldest
first) and horizon h, and returns an array of h point forecasts on the original
(popularity_percent) scale."""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402

from data import BACKEND  # noqa: E402

sys.path.insert(0, BACKEND)
warnings.filterwarnings("ignore")

from statsmodels.tsa.arima.model import ARIMA  # noqa: E402
from statsmodels.tsa.forecasting.theta import ThetaModel  # noqa: E402
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing  # noqa: E402

from app.services import forecast as appf  # the shipped pipeline  # noqa: E402

FLOOR = 1e-9


def m_naive(y, h):
    return np.full(h, y[-1])


def m_mean3(y, h):
    return np.full(h, y[-3:].mean())


def m_drift(y, h):
    slope = (y[-1] - y[0]) / (len(y) - 1)
    return np.maximum(y[-1] + slope * np.arange(1, h + 1), 0.0)


def _log(y):
    return np.log(np.maximum(y, FLOOR))


def _damped_loglin(y, h, window, phi):
    """OLS slope over the last `window` log-values, damped by phi."""
    w = min(window, len(y))
    ly = _log(y[-w:])
    x = np.arange(w)
    slope = np.polyfit(x, ly, 1)[0]
    anchor = ly[-1]
    cum = np.cumsum(phi ** np.arange(1, h + 1))
    return np.exp(anchor + slope * cum)


def m_loglin10_p85(y, h):
    return _damped_loglin(y, h, 10, 0.85)


def m_loglin7_p80(y, h):
    return _damped_loglin(y, h, 7, 0.80)


def m_loglin15_p90(y, h):
    return _damped_loglin(y, h, 15, 0.90)


def m_loglin10_p70(y, h):
    return _damped_loglin(y, h, 10, 0.70)


def m_loglin10_p100(y, h):
    return _damped_loglin(y, h, 10, 1.0)


def m_ets_damped_log(y, h):
    ly = _log(y)
    try:
        fit = ExponentialSmoothing(
            ly, trend="add", damped_trend=True, initialization_method="estimated"
        ).fit()
        return np.exp(fit.forecast(h))
    except Exception:
        return m_naive(y, h)


def m_ets_damped_lvl(y, h):
    try:
        fit = ExponentialSmoothing(
            y, trend="add", damped_trend=True, initialization_method="estimated"
        ).fit()
        return np.maximum(fit.forecast(h), 0.0)
    except Exception:
        return m_naive(y, h)


def m_ses_log(y, h):
    ly = _log(y)
    try:
        fit = SimpleExpSmoothing(ly, initialization_method="estimated").fit()
        return np.exp(fit.forecast(h))
    except Exception:
        return m_naive(y, h)


def m_theta(y, h):
    try:
        fit = ThetaModel(np.maximum(y, FLOOR), period=1, deseasonalize=False).fit()
        return np.maximum(np.asarray(fit.forecast(h)), 0.0)
    except Exception:
        return m_naive(y, h)


def m_current(y, h):
    """Exactly what the app ships today."""
    proc, logged = appf._preprocess(y)
    model, _ = appf._fit_best_model(proc)
    if model is None:
        return m_naive(y, h)
    try:
        out = appf._forecast(model, logged, h)
    except Exception:
        return m_naive(y, h)
    return np.maximum(out["mean"], 0.0)


def _arima_grid(y, h, d_choices, max_p=2, max_q=2, log_always=False):
    if log_always:
        proc, logged = np.log(np.maximum(y, FLOOR)), "log"
    else:
        proc, lg = appf._preprocess(y)
        logged = "log1p" if lg else None
    n = len(proc)
    best = (np.inf, None)
    for p in range(max_p + 1):
        for d in d_choices:
            for q in range(max_q + 1):
                try:
                    fit = ARIMA(proc, order=(p, d, q)).fit()
                    k = p + q + 1
                    if n - k - 1 <= 0:
                        continue
                    aicc = fit.aic + (2 * k * (k + 1)) / (n - k - 1)
                    if aicc < best[0]:
                        best = (aicc, fit)
                except Exception:
                    continue
    if best[1] is None:
        return m_naive(y, h)
    mean = np.asarray(best[1].get_forecast(steps=h).predicted_mean)
    if logged == "log":
        return np.exp(mean)
    if logged == "log1p":
        return np.maximum(np.expm1(mean), 0.0)
    return np.maximum(mean, 0.0)


def m_arima_d1(y, h):
    """Current pipeline but differencing capped at 1 (no d=2 trend explosion)."""
    d = min(appf._find_optimal_differencing(appf._preprocess(y)[0]), 1)
    return _arima_grid(y, h, (d,))


def m_arima_log_d1(y, h):
    """Always log, d capped at 1."""
    d = min(appf._find_optimal_differencing(np.log(np.maximum(y, FLOOR))), 1)
    return _arima_grid(y, h, (d,), log_always=True)


def m_arima_log_d1_short(y, h):
    """Always log, d<=1, fit on the last 30 years only."""
    return m_arima_log_d1(y[-30:], h)


def m_current_short30(y, h):
    return m_current(y[-30:], h)


def m_ets_damped_log_short(y, h):
    return m_ets_damped_log(y[-30:], h)


def m_ens_median(y, h):
    cand = np.vstack([m_naive(y, h), m_ets_damped_log(y, h), m_arima_log_d1(y, h)])
    return np.median(cand, axis=0)


def m_ens_logmean(y, h):
    cand = np.vstack(
        [_log(m_ets_damped_log(y, h)), _log(m_arima_log_d1(y, h)), _log(m_loglin10_p85(y, h))]
    )
    return np.exp(cand.mean(axis=0))


def _ets_log(y, h, phi=None, window=None):
    yy = y[-window:] if window else y
    ly = _log(yy)
    try:
        model = ExponentialSmoothing(
            ly, trend="add", damped_trend=True, initialization_method="estimated"
        )
        fit = model.fit(damping_trend=phi) if phi else model.fit()
        out = np.exp(fit.forecast(h))
        if not np.all(np.isfinite(out)):
            raise ValueError
        return out
    except Exception:
        return m_naive(y, h)


def m_ets_log_phi80(y, h):
    return _ets_log(y, h, phi=0.80)


def m_ets_log_phi90(y, h):
    return _ets_log(y, h, phi=0.90)


def m_ets_log_w30(y, h):
    return _ets_log(y, h, window=30)


def m_ets_log_w20(y, h):
    return _ets_log(y, h, window=20)


def m_theta_log(y, h):
    try:
        fit = ThetaModel(_log(y), period=1, deseasonalize=False).fit()
        return np.exp(np.asarray(fit.forecast(h)))
    except Exception:
        return m_naive(y, h)


def _shrink(pred, y, h, w):
    """Geometric shrink of a forecast toward the naive (last-value) forecast."""
    return np.exp(w * _log(pred) + (1 - w) * np.log(max(y[-1], FLOOR)))


def m_ets_log_shrunk(y, h):
    return _shrink(_ets_log(y, h), y, h, 0.7)


def m_current_shrunk(y, h):
    return _shrink(m_current(y, h), y, h, 0.5)


def m_ens_dampfam(y, h):
    """Log-space mean of the damped-trend family."""
    cand = np.vstack([_log(_ets_log(y, h)), _log(m_loglin10_p70(y, h)), _log(m_loglin7_p80(y, h))])
    return np.exp(cand.mean(axis=0))


def m_ens_damp_arima(y, h):
    cand = np.vstack(
        [
            _log(_ets_log(y, h)),
            _log(m_loglin10_p70(y, h)),
            _log(np.maximum(m_arima_d1(y, h), FLOOR)),
        ]
    )
    return np.exp(cand.mean(axis=0))


METHODS = {k[2:]: v for k, v in list(globals().items()) if k.startswith("m_")}
