"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from .config import ALLOWED_ORIGINS
from .routes.chat import router as chat_router
from .routes.health import router as health_router
from .routes.names import router as names_router

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="Baby Names Explorer API")

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

app.include_router(health_router, prefix="/api")
app.include_router(names_router, prefix="/api")
app.include_router(chat_router, prefix="/api")
