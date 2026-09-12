import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.forecast import arima as forecast  # noqa: E402


def test_stationarity_handles_constant_series():
    # adfuller/kpss reject a constant series outright; _find_optimal_differencing
    # feeds them differenced series that can easily be constant.
    is_stationary, adf_p, kpss_p = forecast._check_stationarity(np.zeros(30))
    assert is_stationary is False
    assert 0.0 <= adf_p <= 1.0
    assert 0.0 <= kpss_p <= 1.0


def test_fit_best_model_survives_constant_series():
    # Must not raise: a flat history should degrade to "no forecast", not a 500.
    model, params = forecast._fit_best_model(np.full(40, 0.001))
    assert model is None or params is not None


def test_the_pipeline_internals_the_research_harness_drives_stay_reachable():
    """`research/forecasting/methods.py` composes its `current` baseline arm
    out of the pipeline's parts rather than calling `fit_forecast`, so that
    every past round benchmarks the shipped code itself. Driving the same
    sequence here keeps the harness from breaking silently.
    """
    series = np.linspace(0.004, 0.011, 45) * (1 + 0.05 * np.sin(np.arange(45)))

    processed, log_applied = forecast._preprocess(series)
    model, _params = forecast._fit_best_model(processed)
    assert model is not None

    out = forecast._forecast(model, log_applied, 5)
    assert len(out["mean"]) == 5

    assert forecast._find_optimal_differencing(processed) in (0, 1, 2)
