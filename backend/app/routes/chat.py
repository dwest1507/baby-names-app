"""POST /api/chat — natural language question → guarded SQL → phrased answer."""

import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

from .. import database
from ..services import chatbot

logger = logging.getLogger(__name__)

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


class HistoryEntry(BaseModel):
    role: str
    content: str = ""
    sql: str | None = None


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=500)
    history: list[HistoryEntry] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    sql: str | None = None


def _answer(body: ChatRequest) -> ChatResponse:
    history = [entry.model_dump() for entry in body.history]

    try:
        sql = chatbot.generate_sql(body.message, history)
    except chatbot.ChatbotUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception:
        logger.exception("SQL generation failed")
        return ChatResponse(
            answer="I couldn't generate a SQL query. Please try rephrasing your question."
        )

    rows, columns, error = chatbot.execute_safe_sql(sql)
    if error:
        return ChatResponse(
            answer=f"I encountered an error executing the SQL query: {error}", sql=sql
        )

    try:
        answer = chatbot.generate_answer(body.message, sql, rows, columns, history)
    except Exception:
        logger.exception("Answer generation failed")
        return ChatResponse(
            answer="I ran the query but couldn't phrase an answer. Please try again.", sql=sql
        )

    return ChatResponse(answer=answer, sql=sql)


@router.post("/chat")
@limiter.limit("30/minute")
async def chat(request: Request, body: ChatRequest) -> ChatResponse:
    available, problem = database.database_status()
    if not available:
        raise HTTPException(status_code=503, detail=problem)
    # Groq calls are made with the sync client; keep the event loop free
    return await run_in_threadpool(_answer, body)
