"""Resolution and read-only access to the baby names SQLite database.

The database is ~1.1 GB and tracked with Git LFS, so a fresh checkout often
contains only a pointer file. This module detects that and can download a real
copy from a Hugging Face repo when one is configured.
"""

import logging
import os
import re
import sqlite3
import time
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


# The budget a model-written query gets. Long enough that no honest question
# about 145 years of names comes close, short enough that a runaway one frees
# its worker before the next visitor notices. See
# docs/adr/0008-a-resource-budget-for-generated-sql.md.
QUERY_BUDGET_SECONDS = 5.0

# The largest single value such a query may materialise. Names and counts are
# tiny; a megabyte is unreachable by accident and stops `randomblob(1e9)` from
# allocating a gigabyte inside one row, where no row cap can reach it.
MAX_VALUE_BYTES = 1_000_000

# How often SQLite pauses to ask whether it should keep going, measured in
# virtual-machine instructions. Small enough to abort a runaway query promptly,
# large enough that the check costs nothing on an ordinary one.
_BUDGET_CHECK_INSTRUCTIONS = 1000


def connect_for_generated_sql() -> sqlite3.Connection:
    """A read-only connection carrying a time and size budget.

    For SQL the model wrote rather than SQL we wrote. Read-only already makes
    such a query harmless to the data; this makes it harmless to the service,
    which the row cap alone cannot do — a cartesian join or a recursive CTE
    does all its damage before the first row is handed back.
    """
    conn = connect()
    deadline = time.monotonic() + QUERY_BUDGET_SECONDS
    # A truthy return aborts the statement in progress with OperationalError.
    conn.set_progress_handler(lambda: time.monotonic() > deadline, _BUDGET_CHECK_INSTRUCTIONS)
    conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, MAX_VALUE_BYTES)
    return conn


def database_status() -> tuple[bool, str | None]:
    """Return (available, problem_message) without raising."""
    try:
        resolve_database_path()
        return True, None
    except DatabaseUnavailableError as e:
        return False, str(e)
