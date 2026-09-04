"""Both rate-limit tiers, and the key they are counted against.

Two tiers, keyed the same way: a general per-visitor ceiling across the whole
API, enforced in ``main.require_shared_secret`` right after the shared secret is
checked, and a tighter per-visitor limit on the endpoint that spends money,
attached to the chat route as a decorator.

They live here rather than in main.py so routes can attach limits without
importing the app (which would be circular). Both must use the same key function
and the same storage, or the limits and the buckets behind them drift apart.
"""

from limits import RateLimitItem, parse_many
from limits.storage import MemoryStorage
from limits.strategies import MovingWindowRateLimiter
from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.requests import Request

FORWARDED_FOR = "X-Forwarded-For"

# The general ceiling: one bucket per visitor shared across every endpoint,
# rather than one bucket per endpoint, so that no route is left unprotected and
# no combination of routes adds up to more than this.
#
# A name search costs a handful of requests (history, forecast, meta), so
# 60/minute leaves room for browsing at speed while stopping a scraper.
GENERAL_LIMITS = list(parse_many("60/minute;1000/hour"))

# Storage is in-process and therefore lost whenever the container scales to zero:
# a sleeping backend wakes with every bucket empty. Accepted deliberately — see
# docs/adr/0002-shared-secret-gateway.md.
_general_storage = MemoryStorage()
_general = MovingWindowRateLimiter(_general_storage)


def visitor_address(request: Request) -> str:
    """The address to rate limit by: the visitor's, not the proxy's.

    Every request arrives from the frontend's server-side proxy, so the direct
    peer address is one rotating platform egress address shared by everybody.
    The visitor's own address is in ``X-Forwarded-For``, which anyone can set —
    so it is trusted only on requests that have already presented the shared
    secret (see ``main.require_shared_secret``, which runs first and marks the
    request). Unverified requests fall back to the direct peer address.
    """
    if getattr(request.state, "secret_verified", False):
        forwarded = request.headers.get(FORWARDED_FOR, "").split(",")[0].strip()
        if forwarded:
            return forwarded
    return get_remote_address(request) or "unknown"


limiter = Limiter(key_func=visitor_address)


def general_limit_exceeded(key: str) -> RateLimitItem | None:
    """Spend one request from a visitor's general allowance.

    Returns the limit that was hit, or None if the request is within budget.

    This is enforced here rather than through slowapi's application limits
    because slowapi's middleware finds a route's handler by scanning
    ``app.routes``, and FastAPI hides routers registered with ``include_router``
    behind a wrapper object with no ``endpoint`` attribute. Every one of this
    app's routes is registered that way, so limits configured on the middleware
    silently never fire. The per-route decorator (used for chat) is unaffected.
    """
    for limit in GENERAL_LIMITS:
        if not _general.hit(limit, key, "api"):
            return limit
    return None


def reset_limits() -> None:
    """Empty every bucket. For tests; production never resets deliberately."""
    limiter.reset()
    _general_storage.reset()


# The chat tier, sized against the provider's free daily allowance.
#
#   Groq's free tier allows 1,000 model requests per day for this chat model
#   (checked when this was written; re-do the arithmetic if the model changes).
#   One chat turn spends TWO of them: one call translates the question into SQL,
#   a second phrases the query results as an answer. The whole site therefore
#   has room for ~500 chat turns per day, shared by every visitor.
#
#   50 turns/day per visitor caps any one visitor at 100 provider calls, a tenth
#   of the daily allowance, so it takes ten determined visitors to exhaust it.
#   5 turns/minute is far above a person's conversational pace and far below
#   what a script can drain the allowance with.
CHAT_LIMIT = "5/minute;50/day"
