"""POST /api/chat — natural language question → guarded SQL → phrased answer."""

import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from .. import database
from ..limiter import CHAT_LIMIT, limiter
from ..services import chatbot

logger = logging.getLogger(__name__)

router = APIRouter()


# The history is written by the caller, not by us, and only its last few
# entries are ever read (chatbot.HISTORY_CONTEXT). Unbounded, it is the cheapest
# way to spend the provider quota that the per-turn rate limit does not cover:
# one request, arbitrarily many tokens. These bounds are generous next to a real
# conversation and small next to an abusive one.
MAX_HISTORY_ENTRIES = 20
MAX_HISTORY_CHARS = 4000


class HistoryEntry(BaseModel):
    role: str
    content: str = Field(default="", max_length=MAX_HISTORY_CHARS)
    sql: str | None = Field(default=None, max_length=MAX_HISTORY_CHARS)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=500)
    history: list[HistoryEntry] = Field(default_factory=list, max_length=MAX_HISTORY_ENTRIES)


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
@limiter.limit(CHAT_LIMIT)
async def chat(request: Request, body: ChatRequest) -> ChatResponse:
    available, problem = database.database_status()
    if not available:
        raise HTTPException(status_code=503, detail=problem)
    # Groq calls are made with the sync client; keep the event loop free
    return await run_in_threadpool(_answer, body)
