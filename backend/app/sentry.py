"""Optional Sentry error reporting.

`SENTRY_DSN` is absent in local development and in CI by design (see
`.env.example` and docs/deployment.md) so that neither ever consumes the free
tier's event budget. This module's only job is to make that absence a
complete no-op: `init_sentry()` returns before touching `sentry_sdk` at all
when no DSN is configured, rather than calling `sentry_sdk.init(dsn=None)`
and relying on the SDK's own handling of that value. That distinction is what
`tests/test_sentry.py` proves, by mocking `sentry_sdk.init` and asserting it
is never called.
"""

import sentry_sdk

from . import config


def init_sentry() -> None:
    """Wire up Sentry if SENTRY_DSN is configured; otherwise do nothing."""
    if not config.SENTRY_DSN:
        return

    sentry_sdk.init(
        dsn=config.SENTRY_DSN,
        environment=config.APP_ENV,
        # No performance tracing or profiling: this deployment only wants
        # exception reporting, and enabling either spends the free tier's
        # (separate, smaller) transaction budget for no benefit here.
        traces_sample_rate=0.0,
        send_default_pii=False,
    )
