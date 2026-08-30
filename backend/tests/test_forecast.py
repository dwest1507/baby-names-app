import numpy as np

from app.services import forecast


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
