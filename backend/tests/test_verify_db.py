"""Tests for scripts/verify_db.py, the pre-deploy artifact check.

A deploy with a missing, truncated, or LFS-pointer database builds and starts
happily -- nothing touches sqlite until the first request -- and then cannot
answer anything. These tests assert the script catches that before deploy
time, against real files rather than mocks: an LFS pointer's actual header
bytes, a real empty file, and a real sqlite database missing a table.
"""

import sqlite3

import pytest


def test_verify_passes_on_a_complete_artifact(tmp_path):
    """A structurally complete artifact certifying deployable scores passes,
    and reports what it found in each table it checked."""
    from scripts.verify_db import verify

    counts = verify(_artifact(tmp_path, "complete.db"))
    assert counts["names"] > 0
    assert counts["forecasts"] > 0
    assert counts["model_evaluation"] > 0


def test_the_sample_database_does_not_clear_the_deploy_gate(sample_db):
    """sample_db (session fixture) is built and has forecasts precomputed
    against it exactly as the real deploy artifact would be -- see
    docs/adr/0004-forecasts-as-a-build-artifact.md -- so everything structural
    about it is right. Nine names cannot earn deployable skill, though, and
    this gate is about quality rather than structure: the sample database is a
    development fixture and publishing one would be the mistake."""
    from scripts.verify_db import VerificationError, verify

    with pytest.raises(VerificationError, match="top100"):
        verify(sample_db)


def test_verify_fails_on_a_missing_file(tmp_path):
    from scripts.verify_db import VerificationError, verify

    with pytest.raises(VerificationError, match="No file was found"):
        verify(str(tmp_path / "does-not-exist.db"))


def test_verify_fails_on_an_empty_file(tmp_path):
    from scripts.verify_db import VerificationError, verify

    path = tmp_path / "empty.db"
    path.write_bytes(b"")
    with pytest.raises(VerificationError, match="not a SQLite database"):
        verify(str(path))


def test_verify_fails_on_an_lfs_pointer(tmp_path):
    from scripts.verify_db import VerificationError, verify

    path = tmp_path / "pointer.db"
    path.write_bytes(
        b"version https://git-lfs.github.com/spec/v1\n"
        b"oid sha256:" + b"0" * 64 + b"\n"
        b"size 1179440640\n"
    )
    with pytest.raises(VerificationError, match="Git LFS pointer"):
        verify(str(path))


def test_verify_fails_on_a_database_missing_the_forecasts_table(tmp_path):
    from scripts.verify_db import VerificationError, verify

    path = tmp_path / "no-forecasts.db"
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE names (name TEXT)")
    conn.execute("INSERT INTO names VALUES ('Emma')")
    conn.commit()
    conn.close()

    with pytest.raises(VerificationError, match="forecasts"):
        verify(str(path))


def test_verify_fails_on_a_database_with_an_empty_table(tmp_path):
    from scripts.verify_db import VerificationError, verify

    path = tmp_path / "truncated.db"
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE names (name TEXT)")
    conn.execute("CREATE TABLE forecasts (name TEXT)")
    conn.execute("CREATE TABLE model_evaluation (tier TEXT)")
    conn.commit()
    conn.close()

    with pytest.raises(VerificationError, match="zero rows"):
        verify(str(path))


# --- The model-quality deploy gate -----------------------------------------

# The scores the real 2025 artifact was measured at (ADR 0010): every tier
# positive, the top 100 comfortably above the pinned floor.
PASSING_SKILLS = {"top100": 0.374, "top1000": 0.238, "top5000": 0.096, "rest": 0.077}


def _artifact(
    tmp_path,
    filename: str,
    *,
    skills: dict[str, float] | None = None,
    span: tuple[int, int, int] = (26, 1995, 2020),
    max_year: int = 2025,
    evaluation_rows: list[tuple] | None = None,
) -> str:
    """A structurally complete artifact carrying the scores it claims.

    Everything `verify` checks before the model gate is real here -- names
    observed through `max_year`, a stored forecast -- so what each test varies
    is only the tier scores the artifact certifies itself with.
    """
    from app import db_schema

    path = tmp_path / filename
    conn = sqlite3.connect(str(path))
    conn.execute(db_schema.CREATE_TABLE)
    conn.executemany(
        "INSERT INTO names VALUES (?, ?, ?, ?, ?, ?)",
        [("Emma", "F", 1000, year, 0.01, 1) for year in range(1880, max_year + 1)],
    )
    conn.execute("CREATE TABLE forecasts (name TEXT, sex TEXT, payload TEXT)")
    conn.execute("INSERT INTO forecasts VALUES ('emma', 'F', '{}')")
    conn.execute(db_schema.CREATE_MODEL_EVALUATION_TABLE)
    if evaluation_rows is None:
        skills = PASSING_SKILLS if skills is None else skills
        evaluation_rows = [(tier, skill, skill, *span) for tier, skill in skills.items()]
    conn.executemany(
        "INSERT INTO model_evaluation VALUES (?, ?, ?, ?, ?, ?)",
        evaluation_rows,
    )
    conn.commit()
    conn.close()
    return str(path)


def test_verify_fails_when_top_100_skill_regresses(tmp_path):
    """The gate the whole table exists for: a build whose top-100 forecasts
    have fallen back toward the ARIMA baseline cannot be published, and the
    message says which tier fell and how far."""
    from scripts.verify_db import VerificationError, verify

    path = _artifact(tmp_path, "regressed.db", skills={**PASSING_SKILLS, "top100": 0.21})

    with pytest.raises(VerificationError, match="top100") as excinfo:
        verify(path)
    assert "0.21" in str(excinfo.value)
    assert "0.30" in str(excinfo.value)


def test_verify_fails_on_an_artifact_with_no_model_evaluation_table(tmp_path):
    """An artifact built before the batch learned to certify itself, or by a
    run that died before writing its scores, has nothing to gate on -- and
    "nothing to gate on" must not read as "nothing wrong"."""
    from scripts.verify_db import VerificationError, verify

    path = _artifact(tmp_path, "uncertified.db")
    conn = sqlite3.connect(path)
    conn.execute("DROP TABLE model_evaluation")
    conn.commit()
    conn.close()

    with pytest.raises(VerificationError, match="model_evaluation"):
        verify(path)


def test_verify_fails_on_an_empty_model_evaluation_table(tmp_path):
    """The table present and empty is the same claim as the table absent."""
    from scripts.verify_db import VerificationError, verify

    path = _artifact(tmp_path, "unscored.db", evaluation_rows=[])

    with pytest.raises(VerificationError, match="zero rows"):
        verify(path)


def test_verify_fails_when_the_top_tier_was_never_scored(tmp_path):
    """ADR 0010: a tier the span never populated is absent rather than zero,
    so a gate can tell "measured, and bad" from "never measured". The top 100
    is the tier the threshold is pinned on -- an artifact that never measured
    it cannot be certified either way, and must not slip through on the rows
    it does carry."""
    from scripts.verify_db import VerificationError, verify

    skills = {tier: skill for tier, skill in PASSING_SKILLS.items() if tier != "top100"}
    path = _artifact(tmp_path, "unmeasured-top.db", skills=skills)

    with pytest.raises(VerificationError, match="top100"):
        verify(path)


def test_verify_fails_when_a_lower_tier_is_no_better_than_no_change(tmp_path):
    """The replacement was for the tail: ARIMA scored -0.211 and -0.413 there,
    which is a forecast that leaves a visitor worse informed than the flat line
    they could have drawn themselves. A build that scores well on the giants
    and at or below zero anywhere below them does not ship, and the message
    names the tier that failed."""
    from scripts.verify_db import VerificationError, verify

    path = _artifact(tmp_path, "flat-tail.db", skills={**PASSING_SKILLS, "top5000": -0.05})

    with pytest.raises(VerificationError, match="top5000") as excinfo:
        verify(path)
    assert "-0.05" in str(excinfo.value)
    assert "top100" not in str(excinfo.value)


def test_verify_fails_when_nothing_below_the_top_tier_was_scored(tmp_path):
    """A build certified on the top 100 alone says nothing about the 24,000
    names below it, which is where the previous pipeline did its damage. An
    artifact carrying no lower tier at all has not been measured against the
    acceptance rule, so it cannot pass it."""
    from scripts.verify_db import VerificationError, verify

    path = _artifact(tmp_path, "top-only.db", skills={"top100": 0.374})

    with pytest.raises(VerificationError, match="nothing below the top 100"):
        verify(path)


def test_verify_fails_on_a_truncated_backtest(tmp_path):
    """A run that died partway leaves scores measured over twelve windows
    presenting themselves beside scores measured over twenty-six. The span is
    stored beside the figures precisely so the gate can tell them apart, and
    the expected span is read off the artifact's own newest observed year
    rather than pinned to a number that goes stale next year."""
    from scripts.verify_db import VerificationError, verify

    path = _artifact(tmp_path, "truncated.db", span=(12, 1995, 2006), max_year=2025)

    with pytest.raises(VerificationError, match="1995:2020") as excinfo:
        verify(path)
    assert "12" in str(excinfo.value)


def test_the_expected_span_advances_with_the_database(tmp_path):
    """Next year's database closes one more five-year window, so a complete
    backtest against it covers twenty-seven origins (1995:2021). The gate has
    to expect that without being edited, or the first thing a 2026 rebuild
    would do is pass a check that had quietly stopped meaning anything."""
    from scripts.verify_db import VerificationError, verify

    stale = _artifact(tmp_path, "stale-span.db", span=(26, 1995, 2020), max_year=2026)
    with pytest.raises(VerificationError, match="1995:2021"):
        verify(stale)

    current = _artifact(tmp_path, "advanced-span.db", span=(27, 1995, 2021), max_year=2026)
    assert verify(current)["model_evaluation"] == len(PASSING_SKILLS)


def test_the_command_reports_the_scores_it_certified(tmp_path, monkeypatch, capsys):
    """`make verify-db [DB=path]` still takes the artifact as one positional
    argument and says nothing on stderr when it passes -- and now says what it
    certified, because "OK" alone does not tell a maintainer which scores they
    are about to publish."""
    from scripts import verify_db

    path = _artifact(tmp_path, "cli-ok.db")
    monkeypatch.setattr("sys.argv", ["verify_db.py", path])

    verify_db.main()

    captured = capsys.readouterr()
    assert captured.err == ""
    assert "top100" in captured.out
    assert "0.374" in captured.out
    assert "1995:2020" in captured.out


def test_the_command_exits_non_zero_when_the_gate_fails(tmp_path, monkeypatch, capsys):
    """A failing gate has to stop `make verify-db`, and say why on stderr."""
    from scripts import verify_db

    path = _artifact(tmp_path, "cli-bad.db", skills={**PASSING_SKILLS, "top100": 0.21})
    monkeypatch.setattr("sys.argv", ["verify_db.py", path])

    with pytest.raises(SystemExit) as excinfo:
        verify_db.main()

    assert excinfo.value.code == 1
    assert "top100" in capsys.readouterr().err
