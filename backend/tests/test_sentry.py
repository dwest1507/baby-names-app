"""Sentry initialization is silent with no DSN, and wires up when one is set.

The only two states that matter here: local dev and CI never set SENTRY_DSN,
so init_sentry() must never even attempt to call sentry_sdk.init in that case
(proven with a mock, not just "no exception raised" — an exception-free no-op
and an exception-free real call look identical from the outside). A deployed
backend with SENTRY_DSN configured must actually initialize the SDK with that
DSN.
"""

from unittest.mock import patch

from app import sentry


def test_init_sentry_does_not_touch_the_sdk_without_a_dsn(monkeypatch):
    monkeypatch.setattr("app.config.SENTRY_DSN", None)
    with patch("app.sentry.sentry_sdk.init") as mock_init:
        sentry.init_sentry()
    mock_init.assert_not_called()


def test_init_sentry_initializes_the_sdk_with_a_configured_dsn(monkeypatch):
    monkeypatch.setattr("app.config.SENTRY_DSN", "https://fake@example.ingest.sentry.io/1")
    with patch("app.sentry.sentry_sdk.init") as mock_init:
        sentry.init_sentry()
    mock_init.assert_called_once()
    _, kwargs = mock_init.call_args
    assert kwargs["dsn"] == "https://fake@example.ingest.sentry.io/1"
