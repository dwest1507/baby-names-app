"""Build the deployable names database from SSA data.

This script produces the deployable database artifact: observed rows only,
carrying the indexes the app's queries need, and preserving existing precomputed
forecasts if already present.

Usage: uv run python scripts/build_db.py [source_path] [output_path]
"""

import os
import re
import sqlite3
import sys
import time
import zipfile
from collections.abc import Callable
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from app import db_schema  # noqa: E402

REPO_ROOT = Path(__file__).parent.parent.parent
DEFAULT_SOURCE = REPO_ROOT / "data" / "names.zip"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "names.built.db"


def download_ssa_zip(dest_path: Path) -> Path:
    """Download the official SSA names.zip archive via headless browser automation."""
    dest_path = Path(dest_path)
    download_dir = dest_path.parent
    download_dir.mkdir(parents=True, exist_ok=True)

    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    prefs = {
        "download.default_directory": str(download_dir.resolve()),
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True,
    }
    options.add_experimental_option("prefs", prefs)

    driver = webdriver.Chrome(options=options)
    try:
        driver.execute_cdp_cmd(
            "Page.setDownloadBehavior",
            {"behavior": "allow", "downloadPath": str(download_dir.resolve())},
        )
        driver.get("https://www.ssa.gov/oact/babynames/limits.html")
        time.sleep(2)

        links = driver.find_elements(By.TAG_NAME, "a")
        target_url = None
        for link in links:
            href = link.get_attribute("href")
            if href and "names.zip" in href:
                target_url = href
                break

        if not target_url:
            target_url = "https://www.ssa.gov/oact/babynames/names.zip"

        driver.get(target_url)

        # Wait up to 30s for download to complete
        start = time.time()
        downloaded = download_dir / "names.zip"
        while time.time() - start < 30:
            if downloaded.exists() and not any(download_dir.glob("*.crdownload")):
                break
            time.sleep(0.5)

        if not downloaded.exists():
            raise RuntimeError(f"Download failed: {downloaded} not found after waiting.")

        if downloaded.resolve() != dest_path.resolve():
            downloaded.replace(dest_path)

        return dest_path
    finally:
        driver.quit()


def ingest_ssa_records(source_path: Path) -> pd.DataFrame:
    """Read SSA yobYYYY.txt files from a zip or directory, returning observed rows.

    Computes popularity_percent and popularity_rank per (sex, year).
    """
    source_path = Path(source_path)
    frames = []

    if source_path.is_file() and source_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(source_path, "r") as zf:
            for item in zf.namelist():
                match = re.search(r"yob(\d{4})\.txt$", item, re.IGNORECASE)
                if match:
                    year = int(match.group(1))
                    with zf.open(item) as f:
                        df_year = pd.read_csv(f, header=None, names=["name", "sex", "total_count"])
                        df_year["year"] = year
                        frames.append(df_year)
    elif source_path.is_dir():
        for file in source_path.iterdir():
            match = re.search(r"yob(\d{4})\.txt$", file.name, re.IGNORECASE)
            if match:
                year = int(match.group(1))
                df_year = pd.read_csv(file, header=None, names=["name", "sex", "total_count"])
                df_year["year"] = year
                frames.append(df_year)
    else:
        raise ValueError(f"Unsupported source format: {source_path}")

    if not frames:
        raise ValueError(f"No yobYYYY.txt files found in source: {source_path}")

    df = pd.concat(frames, ignore_index=True)
    df["year"] = df["year"].astype(int)
    df["total_count"] = df["total_count"].astype(int)

    # Observed rows only: counts must be positive (ADR 0003)
    df = df[df["total_count"] > 0].reset_index(drop=True)
    df = df.drop_duplicates(subset=["name", "sex", "year"])

    # Popularity percent and rank per sex and year
    totals = df.groupby(["sex", "year"])["total_count"].transform("sum")
    df["popularity_percent"] = df["total_count"] / totals
    df["popularity_rank"] = (
        df.groupby(["sex", "year"])["popularity_percent"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    return df[
        ["name", "sex", "total_count", "year", "popularity_percent", "popularity_rank"]
    ].sort_values(by=["year", "sex", "popularity_rank"])


def build(
    source: str | Path | None = None,
    output: str | Path | None = None,
    download_if_missing: bool = True,
    downloader: Callable[[Path], Path] | None = None,
) -> dict:
    """Write the observed, indexed database and return summary metrics."""
    source_path = Path(source) if source is not None else DEFAULT_SOURCE
    output_path = Path(output) if output is not None else DEFAULT_OUTPUT

    if not source_path.exists():
        if download_if_missing:
            download_fn = downloader or download_ssa_zip
            print(f"Source {source_path} not found. Downloading SSA data...")
            download_fn(source_path)
        else:
            raise FileNotFoundError(f"Source file not found: {source_path}")

    started = time.monotonic()
    df = ingest_ssa_records(source_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build in a temporary database file for atomicity and safety
    tmp_output = output_path.with_suffix(".tmp.db")
    if tmp_output.exists():
        tmp_output.unlink()

    conn = sqlite3.connect(str(tmp_output))
    try:
        conn.execute("PRAGMA journal_mode = OFF")
        conn.execute("PRAGMA synchronous = OFF")
        conn.execute(db_schema.CREATE_TABLE)

        df.to_sql("names", conn, if_exists="append", index=False)
        db_schema.create_indexes(conn)

        conn.execute(db_schema.CREATE_FORECASTS_TABLE)
        conn.execute(db_schema.CREATE_CALIBRATION_TABLE)
        conn.execute(db_schema.CREATE_MODEL_EVALUATION_TABLE)

        forecasts_preserved = 0
        # Preserve forecasts and calibration tables from existing database artifact if present
        if output_path.exists():
            db_uri = f"file:{output_path.resolve()}?mode=ro"
            with sqlite3.connect(db_uri, uri=True) as existing_conn:
                forecasts_preserved = _preserve(
                    existing_conn, conn, "forecasts", db_schema.FORECASTS_COLUMNS, required=3
                )
                _preserve(
                    existing_conn,
                    conn,
                    "calibration",
                    db_schema.CALIBRATION_COLUMNS,
                    required=len(db_schema.CALIBRATION_COLUMNS),
                )
                # The scores the deploy gate reads have to survive a `names`
                # rebuild for the same reason the forecasts they describe do:
                # otherwise reingesting the source would leave an artifact
                # that still carries forecasts but can no longer say what they
                # scored, and `verify-db` would reject it.
                _preserve(
                    existing_conn,
                    conn,
                    "model_evaluation",
                    db_schema.MODEL_EVALUATION_COLUMNS,
                    required=len(db_schema.MODEL_EVALUATION_COLUMNS),
                )

        conn.execute("ANALYZE")
        conn.commit()
        (rows,) = conn.execute("SELECT COUNT(*) FROM names").fetchone()
    finally:
        conn.close()

    # VACUUM in its own connection
    conn = sqlite3.connect(str(tmp_output))
    try:
        conn.execute("VACUUM")
    finally:
        conn.close()

    # Atomic swap
    os.replace(tmp_output, output_path)

    return {
        "rows": rows,
        "forecasts_preserved": forecasts_preserved,
        "bytes": output_path.stat().st_size,
        "seconds": time.monotonic() - started,
    }


def _preserve(existing_conn, conn, table: str, columns, required: int) -> int:
    """Copy `table` forward from the artifact being rebuilt, where it still fits.

    `build_db` rebuilds `names` only; forecasts and their calibration are a
    separate, much slower build step (ADR 0004), so rerunning the ingestion
    must not silently discard them. But the artifact on disk can predate a
    schema change, so only the columns the old table still has in common with
    the current schema are copied — and if it is missing one of the first
    `required` of them, the table is left empty rather than filled with rows
    that assert something nobody measured. `make precompute-forecasts`
    rebuilds what is skipped.
    """
    have = {row[1] for row in existing_conn.execute(f"PRAGMA table_info({table})")}
    if not have or not set(columns[:required]) <= have:
        return 0
    shared = [column for column in columns if column in have]
    rows = existing_conn.execute(f"SELECT {', '.join(shared)} FROM {table}").fetchall()
    conn.executemany(
        f"INSERT OR REPLACE INTO {table} ({', '.join(shared)}) "
        f"VALUES ({', '.join('?' * len(shared))})",
        rows,
    )
    return len(rows)


def main() -> None:
    source = sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT_SOURCE)
    output = sys.argv[2] if len(sys.argv) > 2 else str(DEFAULT_OUTPUT)
    result = build(source, output)
    print(f"Kept:                {result['rows']:,} observed rows")
    print(f"Forecasts preserved: {result['forecasts_preserved']:,} rows")
    print(f"Wrote:               {output} ({result['bytes'] / 1024 / 1024:.1f} MB)")
    print(f"Took:                {result['seconds']:.1f}s")


if __name__ == "__main__":
    main()
