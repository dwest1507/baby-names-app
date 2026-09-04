"""Resolution and read-only access to the baby names SQLite database.

The database is ~1.1 GB and tracked with Git LFS, so a fresh checkout often
contains only a pointer file. This module detects that and can download a real
copy from a Hugging Face repo when one is configured.
"""

import logging
import os
import re
import sqlite3
from functools import lru_cache

from . import config

logger = logging.getLogger(__name__)

SQLITE_MAGIC = b"SQLite format 3\x00"
LFS_POINTER_MAGIC = b"version https://git-lfs.github.com/spec/v1"


class DatabaseUnavailableError(RuntimeError):
    """Raised when names.db cannot be resolved to a readable SQLite database."""


def describe_db_problem(path: str) -> str | None:
    """Describe why the file at path is not a usable database, or None if it is."""
    if not os.path.exists(path):
        return f"No file was found at `{path}`."

    with open(path, "rb") as handle:
        head = handle.read(256)

    if head.startswith(LFS_POINTER_MAGIC):
        match = re.search(rb"^size (\d+)$", head, flags=re.MULTILINE)
        expected = f" The real object is {int(match.group(1)) / 1e9:.2f} GB." if match else ""
        return (
            f"`{path}` is a Git LFS pointer of {os.path.getsize(path)} bytes rather than the "
            f"database itself.{expected} Check the repository out with Git LFS, or configure "
            "a remote source so the database is downloaded at startup."
        )

    if not head.startswith(SQLITE_MAGIC):
        return f"`{path}` is not a SQLite database - its file header is unrecognized."

    return None


@lru_cache(maxsize=1)
def resolve_database_path() -> str:
    """Find a readable copy of names.db, downloading it when a remote source is configured."""
    local_path = config.NAMES_DB_PATH
    problem = describe_db_problem(local_path)
    if problem is None:
        return local_path

    if not config.NAMES_DB_REPO:
        raise DatabaseUnavailableError(
            f"{problem}\n\nEither provide a real database at `{local_path}`, point "
            "`NAMES_DB_PATH` at a local copy, or set `NAMES_DB_REPO` so the database is "
            "downloaded at startup."
        )

    try:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            repo_id=config.NAMES_DB_REPO,
            filename=config.NAMES_DB_FILE,
            repo_type=config.NAMES_DB_REPO_TYPE,
            token=config.HF_TOKEN,
        )
    except Exception as e:
        raise DatabaseUnavailableError(
            f"{problem}\n\nDownloading `{config.NAMES_DB_FILE}` from "
            f"`{config.NAMES_DB_REPO}` failed: {e}"
        ) from e

    problem = describe_db_problem(downloaded)
    if problem is not None:
        raise DatabaseUnavailableError(f"The downloaded database is unusable. {problem}")

    logger.info("Using baby names database downloaded from %s", config.NAMES_DB_REPO)
    return downloaded


def connect() -> sqlite3.Connection:
    """Open a read-only connection to the baby names database."""
    path = resolve_database_path()
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def database_status() -> tuple[bool, str | None]:
    """Return (available, problem_message) without raising."""
    try:
        resolve_database_path()
        return True, None
    except DatabaseUnavailableError as e:
        return False, str(e)
