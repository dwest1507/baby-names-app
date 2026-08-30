import logging

from fastapi import APIRouter

from .. import database

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
async def health() -> dict:
    available, problem = database.database_status()
    if not available:
        logger.error("Database unavailable: %s", problem)
    return {"status": "ok", "database": "ok" if available else "unavailable"}
