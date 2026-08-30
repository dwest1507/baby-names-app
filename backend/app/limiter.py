"""The single rate limiter shared by the app and its routes.

It lives here rather than in main.py so routes can attach limits without
importing the app (which would be circular). Both the middleware registered in
main.py and the per-route decorators must use this same instance, or the limits
and the storage backing them drift apart.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
