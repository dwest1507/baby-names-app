from app.services.chatbot import execute_safe_sql, validate_sql_query


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
