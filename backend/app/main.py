"""FastAPI application entry point."""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from .config import ALLOWED_ORIGINS
from .limiter import general_limit_exceeded, limiter, visitor_address
from .routes.chat import router as chat_router
from .routes.health import router as health_router
from .routes.names import router as names_router
from .security import secret_is_valid

app = FastAPI(title="Baby Names Explorer API")

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# The one path the platform probes, and so the one path that cannot present a
# secret. Everything else is guarded uniformly below.
HEALTH_PATH = "/api/health"


# The secret is required uniformly rather than per route, so that no allow-list
# of exceptions can rot as endpoints are added.
@app.middleware("http")
async def require_shared_secret(request: Request, call_next):
    if request.url.path == HEALTH_PATH:
        # Open, and never rate limited: the platform's probe cannot present a
        # secret, and a throttled probe reads as a failed deploy.
        return await call_next(request)

    if not secret_is_valid(request):
        return JSONResponse({"detail": "Unauthorized"}, status_code=401)
    # Only now may the rate limiter believe the forwarded client address.
    request.state.secret_verified = True

    exceeded = general_limit_exceeded(visitor_address(request))
    if exceeded is not None:
        return JSONResponse({"detail": f"Rate limit exceeded: {exceeded}"}, status_code=429)

    return await call_next(request)


# CORS. Kept as defence in depth, but it is NOT what guards this backend: no
# browser request ever reaches it (every call arrives from the frontend's
# server-side proxy), so an origin list has nothing to allow or deny here. The
# shared secret above is the guard. See docs/adr/0002-shared-secret-gateway.md.
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

app.include_router(health_router, prefix="/api")
app.include_router(names_router, prefix="/api")
app.include_router(chat_router, prefix="/api")
