"""Tests for scripts/build_db.py: the automated SSA data ingestion pipeline."""

import sqlite3
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app import db_schema  # noqa: E402


@pytest.fixture
def sample_ssa_zip(tmp_path: Path) -> Path:
    """Create a minimal SSA names.zip archive for testing."""
    zip_path = tmp_path / "names.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(
            "yob2023.txt",
            "Emma,F,1000\nOlivia,F,500\nLiam,M,800\n",
        )
        zf.writestr(
            "yob2024.txt",
            "Emma,F,1200\nSophia,F,600\nLiam,M,900\nNoah,M,100\n",
        )
    return zip_path


def test_build_db_from_local_zip_populates_observed_rows(sample_ssa_zip: Path, tmp_path: Path):
    """build() with a local zip ingests observed records with percent and rank."""
    from scripts.build_db import build

    out_db = tmp_path / "names.built.db"
    result = build(source=sample_ssa_zip, output=out_db)

    assert out_db.exists()
    assert result["rows"] == 7

    conn = sqlite3.connect(str(out_db))
    try:
        # Verify schema
        cursor = conn.execute("PRAGMA table_info(names)")
        cols = {row[1]: row[2] for row in cursor.fetchall()}
        assert cols == {
            "name": "TEXT",
            "sex": "TEXT",
            "total_count": "INTEGER",
            "year": "INTEGER",
            "popularity_percent": "REAL",
            "popularity_rank": "INTEGER",
        }

        # Verify Emma in 2023: 1000 out of 1500 F births => percent ~0.6667, rank 1
        row = conn.execute(
            "SELECT total_count, popularity_percent, popularity_rank FROM names "
            "WHERE name = 'Emma' AND sex = 'F' AND year = 2023"
        ).fetchone()
        assert row is not None
        count, pct, rank = row
        assert count == 1000
        assert pytest.approx(pct, rel=1e-3) == 1000 / 1500
        assert rank == 1

        # Verify Olivia in 2023: 500 out of 1500 F births => rank 2
        row = conn.execute(
            "SELECT total_count, popularity_percent, popularity_rank FROM names "
            "WHERE name = 'Olivia' AND sex = 'F' AND year = 2023"
        ).fetchone()
        assert row is not None
        count, pct, rank = row
        assert count == 500
        assert pytest.approx(pct, rel=1e-3) == 500 / 1500
        assert rank == 2

        # Verify no zero-count rows exist
        (zeros,) = conn.execute("SELECT COUNT(*) FROM names WHERE total_count <= 0").fetchone()
        assert zeros == 0
    finally:
        conn.close()


def test_build_db_creates_all_canonical_indexes(sample_ssa_zip: Path, tmp_path: Path):
    """build() creates all canonical tables and indexes from db_schema."""
    from scripts.build_db import build

    out_db = tmp_path / "names.built.db"
    build(source=sample_ssa_zip, output=out_db)

    conn = sqlite3.connect(str(out_db))
    try:
        tables = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        assert "names" in tables
        assert "forecasts" in tables
        assert "calibration" in tables

        indexes = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'")
        }
        assert "idx_names_lower_name_sex_year" in indexes
        assert "idx_names_sex_year" in indexes
        assert "idx_names_name_sex_year" in indexes
    finally:
        conn.close()


def test_build_db_preserves_existing_forecasts(sample_ssa_zip: Path, tmp_path: Path):
    """Rebuilding observed rows preserves any existing rows in forecasts and calibration."""
    from scripts.build_db import build

    out_db = tmp_path / "names.built.db"

    # Pre-create database with existing forecasts and calibration
    conn = sqlite3.connect(str(out_db))
    conn.execute(
        "CREATE TABLE forecasts (name TEXT NOT NULL, sex TEXT NOT NULL, payload TEXT NOT NULL, "
        "coverage_hits TEXT, coverage_n TEXT, PRIMARY KEY (name, sex))"
    )
    conn.execute(
        "INSERT INTO forecasts VALUES "
        "('Emma', 'F', '{\"test\": 123}', '{\"0.8\": 4}', '{\"0.8\": 5}')"
    )
    conn.execute(db_schema.CREATE_CALIBRATION_TABLE)
    conn.execute("INSERT INTO calibration VALUES (0.8, 'top100', 1, 0.81, 100)")
    conn.commit()
    conn.close()

    result = build(source=sample_ssa_zip, output=out_db)

    assert result["forecasts_preserved"] == 1

    conn = sqlite3.connect(str(out_db))
    try:
        # Check forecasts row preserved
        row = conn.execute(
            "SELECT name, sex, payload FROM forecasts WHERE name = 'Emma'"
        ).fetchone()
        assert row is not None
        assert row[0] == "Emma"
        assert row[1] == "F"
        assert '{"test": 123}' in row[2]

        # Check calibration row preserved, stratum and all
        cal = conn.execute(
            "SELECT empirical_coverage, n FROM calibration "
            "WHERE nominal_level = 0.8 AND tier = 'top100' AND volatility_bin = 1"
        ).fetchone()
        assert cal is not None
        assert cal[0] == 0.81
        assert cal[1] == 100

        # Check names table also populated
        (name_count,) = conn.execute("SELECT COUNT(*) FROM names").fetchone()
        assert name_count == 7
    finally:
        conn.close()


def test_build_db_drops_a_calibration_table_the_batch_can_no_longer_read(
    sample_ssa_zip: Path, tmp_path: Path
):
    """Coverage measured before it was measured per stratum cannot be carried over.

    A pre-#44 artifact's `calibration` rows are keyed by nominal level alone,
    so there is no stratum to file them under and no honest way to guess one —
    a row that claimed to describe `top100` would be asserting something
    nobody measured. They are dropped, and the next
    `make precompute-forecasts` rebuilds the table from `forecasts`, which is
    where the per-name coverage that feeds it actually lives.
    """
    from scripts.build_db import build

    out_db = tmp_path / "names.built.db"
    conn = sqlite3.connect(str(out_db))
    conn.execute(
        "CREATE TABLE calibration (nominal_level REAL NOT NULL PRIMARY KEY, "
        "empirical_coverage REAL NOT NULL, n INTEGER NOT NULL)"
    )
    conn.execute("INSERT INTO calibration VALUES (0.8, 0.44, 45)")
    conn.commit()
    conn.close()

    build(source=sample_ssa_zip, output=out_db)

    conn = sqlite3.connect(str(out_db))
    try:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(calibration)")}
        (remaining,) = conn.execute("SELECT COUNT(*) FROM calibration").fetchone()
    finally:
        conn.close()

    assert columns == set(db_schema.CALIBRATION_COLUMNS)
    assert remaining == 0


def test_build_db_raises_if_source_missing_and_download_disabled(tmp_path: Path):
    """build() raises FileNotFoundError if source is missing and download_if_missing=False."""
    from scripts.build_db import build

    missing_source = tmp_path / "nonexistent.zip"
    out_db = tmp_path / "names.built.db"

    with pytest.raises(FileNotFoundError, match="Source file not found"):
        build(source=missing_source, output=out_db, download_if_missing=False)


def test_build_db_invokes_downloader_when_source_missing(tmp_path: Path):
    """build() triggers downloader when source zip is missing and succeeds."""
    from scripts.build_db import build

    missing_source = tmp_path / "downloaded_names.zip"
    out_db = tmp_path / "names.built.db"
    downloader_called = False

    def fake_downloader(dest: Path) -> Path:
        nonlocal downloader_called
        downloader_called = True
        with zipfile.ZipFile(dest, "w") as zf:
            zf.writestr("yob2024.txt", "Liam,M,500\nOlivia,F,400\n")
        return dest

    result = build(
        source=missing_source,
        output=out_db,
        download_if_missing=True,
        downloader=fake_downloader,
    )

    assert downloader_called is True
    assert missing_source.exists()
    assert result["rows"] == 2
