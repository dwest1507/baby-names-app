"""Groq-powered natural-language chatbot over the names database.

Two LLM calls per question: one to translate the question into a guarded
SELECT query, one to phrase the query results as an answer.
"""

import re
import sqlite3

from groq import Groq

from .. import config, database

MAX_ROWS = 1000
MAX_RESULT_CHARS = 5000
HISTORY_CONTEXT = 4

SCHEMA_CONTEXT = """
Database Schema:
- Table: names
- Columns:
  - name (TEXT): Baby name
  - sex (TEXT): 'M' for Male or 'F' for Female
  - total_count (INTEGER): Number of babies with this name in the given year
  - year (INTEGER): Year (1880-2024)
  - popularity_percent (REAL): Fraction (NOT a percentage) of babies of that sex born that
    year who were given this name. Multiply by 100 for a percentage: 0.011 means 1.1%.
  - popularity_rank (INTEGER): Ranking of the name for the given sex/year (1 = most popular)

Sparsity (important):
- The table is sparse. A row exists only for a name/sex/year with at least 5 recorded
  births: the Social Security Administration suppresses smaller counts for privacy, so
  there is no row for every name in every year.
- Always filter with `total_count > 0`. A zero count is fabricated padding, not an
  observation.
- A missing row means "fewer than 5, or none". It is not zero and must never be reported
  as "0 babies" — say instead that the name was not recorded that year.
- Per-year averages must be taken over the years present for that name, never divided by
  a full span of years, or the answer is diluted by years that carry no data.

Important Guidelines:
- Prefer aggregation queries with GROUP BY, SUM, COUNT, AVG, etc. when summarizing data
- Always include a LIMIT clause (max 1000 rows)
- Use appropriate WHERE clauses to filter data
- For name searches, use LOWER() function for case-insensitive matching
"""

SQL_SYSTEM_PROMPT = f"""You are a SQL query generator. Your task is to translate natural
language questions about a baby names database into SQL queries.

{SCHEMA_CONTEXT}

Rules:
1. Only generate SELECT queries
2. Always include LIMIT 1000 in your queries
3. Prefer using aggregations (GROUP BY, SUM, COUNT, AVG, MAX, MIN) to summarize data
   rather than returning large result sets
4. Return only the SQL query, no explanation or markdown formatting
5. Use proper SQL syntax for SQLite
"""

ANSWER_SYSTEM_PROMPT = """You are a helpful assistant that answers questions about baby
names data from the Social Security Administration database.
You analyze SQL query results and provide clear, concise answers to user questions.
Always format numbers nicely (e.g., use commas for large numbers).
The data is sparse: a year with no row for a name means fewer than 5 births were
recorded, or none — never say "0 babies" for such a year, say the name was not
recorded that year. Averages per year are taken over the years present in the results.
`popularity_percent` is a fraction, not a percentage: multiply it by 100 before reporting
it as a percentage (0.011 is 1.1%).
Be conversational and helpful, but stay accurate to the data.
"""


class ChatbotUnavailableError(RuntimeError):
    """Raised when the chatbot is not configured (missing GROQ_API_KEY)."""


def get_client() -> Groq:
    if not config.GROQ_API_KEY:
        raise ChatbotUnavailableError(
            "GROQ_API_KEY is not configured, so the AI chatbot is unavailable."
        )
    return Groq(api_key=config.GROQ_API_KEY)


def validate_sql_query(query: str) -> tuple[bool, str]:
    """Validate SQL query for safety — only SELECT statements are allowed."""
    if not query:
        return False, "Empty query"

    query_upper = query.strip().upper()

    if not query_upper.startswith("SELECT"):
        return False, "Only SELECT queries are allowed"

    dangerous = [
        "DROP",
        "DELETE",
        "INSERT",
        "UPDATE",
        "ALTER",
        "CREATE",
        "TRUNCATE",
        "EXEC",
        "EXECUTE",
        "PRAGMA",
        "ATTACH",
        "DETACH",
    ]
    for keyword in dangerous:
        if re.search(rf"\b{keyword}\b", query_upper):
            return False, f"Query contains forbidden keyword: {keyword}"

    if "LIMIT" not in query_upper:
        if query.rstrip().endswith(";"):
            query = query.rstrip()[:-1].rstrip() + f" LIMIT {MAX_ROWS};"
        else:
            query = query.rstrip() + f" LIMIT {MAX_ROWS}"

    return True, query


def execute_safe_sql(query: str) -> tuple[list[dict] | None, list[str] | None, str | None]:
    """Execute a validated SELECT on a read-only connection with a row cap.

    Returns (rows, columns, error).
    """
    is_valid, result = validate_sql_query(query)
    if not is_valid:
        return None, None, result
    query = result

    limit_match = re.search(r"LIMIT\s+(\d+)", query, flags=re.IGNORECASE)
    if limit_match and int(limit_match.group(1)) > MAX_ROWS:
        query = re.sub(r"LIMIT\s+\d+", f"LIMIT {MAX_ROWS}", query, flags=re.IGNORECASE)

    try:
        conn = database.connect()
        try:
            cursor = conn.execute(query)
            columns = [c[0] for c in cursor.description]
            rows = [dict(zip(columns, row, strict=True)) for row in cursor.fetchmany(MAX_ROWS)]
        finally:
            conn.close()
        return rows, columns, None
    except sqlite3.Error as e:
        return None, None, str(e)


def _strip_code_fences(text: str) -> str:
    text = re.sub(r"^```sql\s*", "", text.strip())
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()


def generate_sql(question: str, history: list[dict]) -> str:
    """Translate a natural language question into a SQL query via Groq."""
    client = get_client()

    messages = [{"role": "system", "content": SQL_SYSTEM_PROMPT}]
    for entry in history[-HISTORY_CONTEXT:]:
        if entry.get("role") == "user":
            messages.append({"role": "user", "content": entry["content"]})
        elif entry.get("role") == "assistant" and entry.get("sql"):
            messages.append({"role": "assistant", "content": f"SQL: {entry['sql']}"})
    messages.append({"role": "user", "content": question})

    response = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=messages,
        temperature=0.1,
        # gpt-oss is a reasoning model: reasoning tokens count toward this budget
        max_tokens=2000,
    )
    return _strip_code_fences(response.choices[0].message.content)


def _format_rows(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return "No results returned from the query."

    lines = ["\t".join(columns)]
    for row in rows:
        lines.append("\t".join(str(row[c]) for c in columns))
    text = "\n".join(lines)
    if len(text) > MAX_RESULT_CHARS:
        text = text[:MAX_RESULT_CHARS] + "\n... (truncated)"
    return text


def generate_answer(
    question: str, sql: str, rows: list[dict], columns: list[str], history: list[dict]
) -> str:
    """Phrase the SQL results as a natural-language answer via Groq."""
    client = get_client()

    messages = [{"role": "system", "content": ANSWER_SYSTEM_PROMPT}]
    for entry in history[-HISTORY_CONTEXT:]:
        role = entry.get("role")
        if role in ("user", "assistant") and entry.get("content"):
            messages.append({"role": role, "content": entry["content"]})

    messages.append(
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"SQL Query Executed:\n{sql}\n\n"
                f"Query Results:\n{_format_rows(rows, columns)}\n\n"
                "Please answer the user's question based on the query results above. "
                "Be concise and helpful."
            ),
        }
    )

    response = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=messages,
        temperature=0.3,
        # gpt-oss is a reasoning model: reasoning tokens count toward this budget
        max_tokens=3000,
    )
    return response.choices[0].message.content.strip()
