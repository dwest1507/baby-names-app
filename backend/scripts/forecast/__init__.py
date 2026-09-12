"""Offline forecasting batch.

Everything here runs from `scripts/precompute_forecasts.py`, never on the
request path, and is excluded from the production container image — the
Dockerfile copies `app/` only. That is what lets the fitting libraries
(`statsmodels`, `scipy`) stay out of the runtime dependency set. See
docs/adr/0004-forecasts-as-a-build-artifact.md.
"""
