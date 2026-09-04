"""The shared secret the frontend proxy presents on every backend call.

The deployed backend has a public URL and no other authentication, so this is
the only thing standing between a stranger and the metered chat endpoint.
"""

import hmac

from starlette.requests import Request

from . import config

SECRET_HEADER = "X-Backend-Secret"


def secret_is_valid(request: Request) -> bool:
    """True when the request carries the configured secret.

    Read from the config module at call time (rather than importing the value)
    so tests and a restarted process both see the current setting.
    """
    expected = config.BACKEND_SHARED_SECRET
    if not expected:
        # No secret configured. Locally that is normal — `make dev` needs no
        # setup and the backend is reachable only from the developer's machine.
        # Deployed, it means the guard is missing, so nothing is served.
        return not config.IS_PRODUCTION
    return hmac.compare_digest(request.headers.get(SECRET_HEADER, ""), expected)
