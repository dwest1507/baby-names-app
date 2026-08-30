"""Name data endpoints: top names, per-name history, and ARIMA forecasts."""

from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool

from .. import database
from ..services import forecast, queries

router = APIRouter()


def _require_database() -> None:
    available, problem = database.database_status()
    if not available:
        raise HTTPException(status_code=503, detail=problem)


@router.get("/meta")
async def meta() -> dict:
    _require_database()
    return queries.get_year_range()


@router.get("/top-names")
async def top_names(
    sex: Literal["M", "F"] = Query(...),
    year: int = Query(..., ge=1880, le=2100),
    limit: int = Query(20, ge=1, le=100),
) -> dict:
    _require_database()
    return {"names": queries.get_top_names(sex, year, limit)}


@router.get("/names/{name}")
async def name_history(name: str, sex: Literal["M", "F"] = Query(...)) -> dict:
    _require_database()
    history = queries.get_name_history(name, sex)
    if not history:
        raise HTTPException(status_code=404, detail=f"No data found for '{name}' ({sex})")
    return {"name": history[0]["name"], "sex": sex, "history": history}


@router.get("/names/{name}/forecast")
async def name_forecast(name: str, sex: Literal["M", "F"] = Query(...)) -> dict:
    _require_database()
    # ARIMA fitting is CPU-bound; keep the event loop free
    payload = await run_in_threadpool(forecast.forecast_name, name.lower(), sex)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"No data found for '{name}' ({sex})")
    return payload
