"""Groq-powered natural-language chatbot over the names database.

Two LLM calls per question: one to translate the question into a read-only
query, one to phrase the query results as an answer.

The generated query is guarded four ways, because none of them is sufficient
alone: it must read rather than write (``validate_sql_query``), it runs on a
read-only connection, it is capped to ``MAX_ROWS`` rows, and it runs under the
time and size budget in ``database.connect_for_generated_sql`` — the only one
of the four that a cartesian join or a recursive CTE respects.
"""

import re
import sqlite3

from groq import Groq

from .. import config, database

MAX_ROWS = 1000
# Verbs that must never reach the database, mapped to the pattern that catches
# them. Redundant three times over - the connection is read-only, the driver
# refuses a second statement, and a leading SELECT or WITH is required - but
# cheap, and it turns a mistake into a clear message instead of a SQLite error.
FORBIDDEN = {
    keyword: rf"\b{keyword}\b"
    for keyword in (
        "DROP",
        "DELETE",
        "INSERT",
        "UPDATE",
        "ALTER",
        "CREATE",
        "TRUNCATE",
        "EXEC",
        "EXECUTE",
        "ATTACH",
        "DETACH",
    )
} | {
    # The table-valued forms - pragma_table_info, pragma_database_list, which
    # reports the file's path on disk - need the trailing \w*. A plain
    # \bPRAGMA\b misses them, because `_` is a word character.
    "PRAGMA": r"\bPRAGMA\w*",
    # Disabled by default in Python's sqlite3, so this is a guard against that
    # default ever changing rather than against today's behaviour.
    "LOAD_EXTENSION": r"\bLOAD_EXTENSION\b",
}

BUDGET_EXCEEDED_MESSAGE = "that query took too long to run - please ask something narrower"
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
- Use appropriate WHERE clauses to filter data
- For name searches, use LOWER() function for case-insensitive matching
"""

SQL_SYSTEM_PROMPT = f"""You are a SQL query generator. Your task is to translate natural
language questions about a baby names database into SQL queries.

{SCHEMA_CONTEXT}

Rules:
1. Generate read-only queries only: a query must begin with SELECT, or with WITH for a
   CTE. Never INSERT, UPDATE, DELETE, CREATE, ALTER, DROP, ATTACH or PRAGMA.
2. CTEs are welcome. Prefer `WITH ranked AS (...) SELECT ...` over a repeated subquery
   when a question needs two stages, such as ranking and then filtering.
3. A LIMIT of {MAX_ROWS} rows is applied to your query automatically, so you need not add
   one. Add your own smaller LIMIT when the question asks for a specific number of rows
   ("the top 5"), and it will be respected.
4. Never write a recursive CTE (WITH RECURSIVE) or a cross join. Queries are cancelled
   after a few seconds, and those are the two that never finish.
5. Prefer using aggregations (GROUP BY, SUM, COUNT, AVG, MAX, MIN) to summarize data
   rather than returning large result sets
6. Return only the SQL query, no explanation or markdown formatting
7. Use proper SQL syntax for SQLite
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


def _bounded(query: str) -> str:
    """Wrap a query so the row cap is the outermost thing it does.

    Reading LIMIT out of the query text cannot work: the model's LIMIT may sit
    in a subquery, where rewriting it answers a different question, and a
    trailing `-- comment` both hides a cap that is there and swallows one
    appended to the end. Wrapping needs to know none of that.
    """
    inner = query.strip().rstrip(";").rstrip()
    # The newline is load-bearing: without it a trailing `--` comment would
    # swallow the closing parenthesis.
    return f"SELECT * FROM (\n{inner}\n) LIMIT {MAX_ROWS}"


def validate_sql_query(query: str) -> tuple[bool, str]:
    """Check that a query only reads, and wrap it in the row cap.

    Returns ``(True, query_to_run)`` — the wrapped query, not the one passed in
    — or ``(False, reason)``, where the reason is shown to whoever asked the
    question.
    """
    if not query:
        return False, "Empty query"

    query_upper = query.strip().upper()

    # A leading WITH is a read: SQLite also accepts WITH ... INSERT/UPDATE/DELETE,
    # but those verbs are refused below and the connection is read-only besides.
    if not re.match(r"(SELECT|WITH)\b", query_upper):
        return False, "Only read-only SELECT or WITH queries are allowed"

    for keyword, pattern in FORBIDDEN.items():
        if re.search(pattern, query_upper):
            return False, f"Query contains forbidden keyword: {keyword}"

    return True, _bounded(query)


def execute_safe_sql(query: str) -> tuple[list[dict] | None, list[str] | None, str | None]:
    """Execute a validated read on a budgeted, read-only connection.

    Returns (rows, columns, error).
    """
    is_valid, result = validate_sql_query(query)
    if not is_valid:
        return None, None, result
    query = result

    try:
        conn = database.connect_for_generated_sql()
        try:
            cursor = conn.execute(query)
            columns = [c[0] for c in cursor.description]
            rows = [dict(zip(columns, row, strict=True)) for row in cursor.fetchmany(MAX_ROWS)]
        finally:
            conn.close()
        return rows, columns, None
    except sqlite3.OperationalError as e:
        # The budget in database.connect_for_generated_sql fires as a bare
        # "interrupted", which tells whoever asked the question nothing.
        if "interrupted" in str(e):
            return None, None, BUDGET_EXCEEDED_MESSAGE
        return None, None, str(e)
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
