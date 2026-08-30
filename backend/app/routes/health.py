from fastapi import APIRouter

from .. import database

router = APIRouter()


@router.get("/health")
async def health() -> dict:
    available, problem = database.database_status()
    return {"status": "ok", "database": "ok" if available else "unavailable", "detail": problem}
