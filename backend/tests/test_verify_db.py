"""Tests for scripts/verify_db.py, the pre-deploy artifact check.

A deploy with a missing, truncated, or LFS-pointer database builds and starts
happily -- nothing touches sqlite until the first request -- and then cannot
answer anything. These tests assert the script catches that before deploy
time, against real files rather than mocks: an LFS pointer's actual header
bytes, a real empty file, and a real sqlite database missing a table.
"""

import sqlite3

import pytest


def test_verify_passes_on_a_complete_database(sample_db):
    """sample_db (session fixture) is built and has forecasts precomputed
    against it exactly as the real deploy artifact would be -- see
    docs/adr/0004-forecasts-as-a-build-artifact.md."""
    from scripts.verify_db import verify

    counts = verify(sample_db)
    assert counts["names"] > 0
    assert counts["forecasts"] > 0


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
    conn.commit()
    conn.close()

    with pytest.raises(VerificationError, match="zero rows"):
        verify(str(path))
