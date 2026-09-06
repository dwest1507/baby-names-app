import time

from app.services.chatbot import MAX_ROWS, execute_safe_sql, validate_sql_query


def test_rejects_non_select():
    ok, message = validate_sql_query("DROP TABLE names")
    assert not ok
    assert "SELECT" in message


def test_rejects_forbidden_keyword():
    ok, message = validate_sql_query("SELECT * FROM names; DELETE FROM names")
    assert not ok
    assert "DELETE" in message


def test_allows_column_names_containing_keyword_substrings():
    # "created_at" contains CREATE as a substring but not as a word
    ok, _ = validate_sql_query("SELECT name AS created_at FROM names LIMIT 5")
    assert ok


def test_adds_limit_when_missing():
    ok, query = validate_sql_query("SELECT name FROM names")
    assert ok
    assert "LIMIT 1000" in query


def test_execute_caps_limit():
    rows, columns, error = execute_safe_sql("SELECT * FROM names LIMIT 999999")
    assert error is None
    assert columns is not None
    assert len(rows) <= 1000


def test_execute_returns_rows():
    rows, columns, error = execute_safe_sql(
        "SELECT name, total_count FROM names WHERE sex = 'M' AND year = 2020 "
        "ORDER BY total_count DESC LIMIT 3"
    )
    assert error is None
    assert columns == ["name", "total_count"]
    assert len(rows) == 3


def test_execute_surfaces_sql_errors():
    rows, columns, error = execute_safe_sql("SELECT nope FROM missing_table LIMIT 5")
    assert rows is None
    assert error is not None


def test_write_rejected_by_readonly_connection():
    # Belt and braces: even if validation were bypassed, the connection is read-only
    from app import database

    conn = database.connect()
    try:
        try:
            conn.execute("INSERT INTO names VALUES ('X', 'M', 1, 2024, 0.1, 1)")
            raised = False
        except Exception:
            raised = True
        assert raised
    finally:
        conn.close()


def test_schema_context_describes_suppression_and_sparsity():
    from app.services.chatbot import SCHEMA_CONTEXT

    text = SCHEMA_CONTEXT.lower()
    # The table is sparse, and every row that exists is an observation of at
    # least five births — the source suppresses anything smaller.
    assert "at least 5" in text
    assert "missing row" in text
    assert "fewer than 5, or none" in text
    assert "not zero" in text
    # Averages must be taken over the years present, not a padded year range.
    assert "years present" in text


def test_allows_common_table_expression():
    # CTEs are how the model naturally writes rank-then-filter and
    # year-over-year questions; a read-only connection makes them no more
    # dangerous than the equivalent subquery.
    rows, columns, error = execute_safe_sql(
        "WITH busiest AS ("
        "  SELECT name, SUM(total_count) AS births FROM names"
        "  WHERE sex = 'M' GROUP BY name ORDER BY births DESC LIMIT 3"
        ") SELECT name, births FROM busiest"
    )
    assert error is None
    assert columns == ["name", "births"]
    assert len(rows) == 3


def test_rejects_write_disguised_as_a_cte():
    # SQLite accepts WITH ... DELETE. Allowing a leading WITH must not open it.
    ok, message = validate_sql_query("WITH doomed AS (SELECT name FROM names) DELETE FROM names")
    assert not ok
    assert "DELETE" in message


def test_runaway_query_is_aborted(monkeypatch):
    # An unbounded recursive CTE never finishes on its own. Neither LIMIT nor
    # fetchmany bounds it: the work happens before a single row is handed back.
    from app import database

    monkeypatch.setattr(database, "QUERY_BUDGET_SECONDS", 0.2)

    started = time.monotonic()
    rows, columns, error = execute_safe_sql(
        "WITH RECURSIVE forever(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM forever) "
        "SELECT COUNT(*) FROM forever"
    )
    elapsed = time.monotonic() - started

    assert rows is None
    assert error is not None
    assert elapsed < 5


def test_aborted_query_explains_itself(monkeypatch):
    # "interrupted" is what SQLite says; it is shown to whoever asked the
    # question, so it has to say what actually happened.
    from app import database

    monkeypatch.setattr(database, "QUERY_BUDGET_SECONDS", 0.2)

    _, _, error = execute_safe_sql(
        "WITH RECURSIVE forever(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM forever) "
        "SELECT COUNT(*) FROM forever"
    )
    assert "too long" in error
    assert "narrower" in error


def test_oversized_value_is_refused():
    # A single row can still exhaust the container's memory. randomblob is one
    # row, so LIMIT and fetchmany are both irrelevant to it.
    rows, _, error = execute_safe_sql("SELECT length(randomblob(900000000))")
    assert rows is None
    assert error is not None


def test_row_cap_does_not_rewrite_an_inner_limit(monkeypatch):
    # The cap belongs to the outermost query. Rewriting a LIMIT the model put
    # inside a subquery silently answers a different question than the one asked.
    monkeypatch.setattr("app.services.chatbot.MAX_ROWS", 3)

    rows, _, error = execute_safe_sql(
        "SELECT COUNT(*) AS n FROM (SELECT name FROM names ORDER BY name LIMIT 5)"
    )
    assert error is None
    assert rows[0]["n"] == 5


def test_trailing_comment_neither_breaks_nor_escapes_the_cap():
    # A trailing line comment used to defeat the cap two ways: the word "limit"
    # inside it looked like a cap that was already there, and a cap appended
    # after it landed inside the comment.
    query = "SELECT name FROM names ORDER BY name -- no limit intended"

    ok, bounded = validate_sql_query(query)
    assert ok
    assert bounded.rstrip().endswith("LIMIT 1000")

    rows, _, error = execute_safe_sql(query)
    assert error is None
    assert rows


def test_ordering_survives_the_row_cap():
    # "Top 5" answers are only correct if the order the model asked for is the
    # order that comes back out of the wrapper.
    rows, _, error = execute_safe_sql(
        "SELECT name, SUM(total_count) AS births FROM names "
        "GROUP BY name ORDER BY births DESC LIMIT 5"
    )
    assert error is None
    births = [row["births"] for row in rows]
    assert births == sorted(births, reverse=True)
    assert len(births) == 5


def test_rejects_pragma_table_valued_functions():
    # `\bPRAGMA\b` does not match `pragma_database_list` - the underscore is a
    # word character - and that function returns the database's path on disk.
    ok, message = validate_sql_query("SELECT * FROM pragma_database_list")
    assert not ok
    assert "PRAGMA" in message.upper()


def test_rejection_message_names_a_keyword_not_a_pattern():
    # The message is shown to whoever asked the question.
    _, message = validate_sql_query("SELECT * FROM pragma_database_list")
    assert "\\" not in message
    assert "*" not in message


def test_rejects_load_extension():
    # Python's sqlite3 disables extension loading by default, so today this
    # fails at execution. The validator should not depend on that default.
    ok, message = validate_sql_query("SELECT load_extension('/tmp/evil.so')")
    assert not ok
    assert "LOAD_EXTENSION" in message


def test_prompt_and_validator_agree_on_what_is_allowed():
    # The prompt is half the guardrail: a validator that accepts CTEs while the
    # prompt forbids them just means the model never writes one.
    from app.services.chatbot import SQL_SYSTEM_PROMPT

    # The validator accepts CTEs, so the prompt has to say so.
    assert "CTE" in SQL_SYSTEM_PROMPT
    assert validate_sql_query("WITH x AS (SELECT 1 AS n) SELECT n FROM x")[0]

    # And the cap is applied for the model, so the prompt must not tell it that
    # writing its own LIMIT is what enforces one.
    assert "Always include LIMIT" not in SQL_SYSTEM_PROMPT
    assert str(MAX_ROWS) in SQL_SYSTEM_PROMPT
