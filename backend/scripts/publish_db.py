"""Publish the built database to a public Hugging Face dataset.

The deployed backend never builds the database itself -- it downloads a copy
baked in at container-build time (see `backend/Dockerfile` and
docs/adr/0006-database-as-published-build-artifact.md). This script is the
other end of that pipeline: it pushes `data/names.built.db` (the artifact
`scripts/build_db.py` and `scripts/precompute_forecasts.py` produce) to a
Hugging Face dataset repo, so the Dockerfile has something to download.

The dataset is created public (no token needed to *read* it -- the underlying
SSA data is public) even though publishing it requires a token, since HF
requires authentication to write regardless of the resulting repo's
visibility.

This is a manual, human-run step: it needs a real `HF_TOKEN` with write
access, which is not available to an agent in this environment. Its logic is
covered by tests that mock `huggingface_hub.HfApi` (see
tests/test_publish_db.py); it has never been run against a real Hugging Face
endpoint.

Usage: HF_TOKEN=... uv run python scripts/publish_db.py <repo_id> [db_path]
  repo_id  e.g. "someuser/baby-names-db"
  db_path  defaults to data/names.built.db
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from huggingface_hub import HfApi  # noqa: E402

REPO_ROOT = Path(__file__).parent.parent.parent
DEFAULT_DB = str(REPO_ROOT / "data" / "names.built.db")

# Fixed so it matches app.config.NAMES_DB_FILE's default, which is what the
# backend asks Hugging Face for.
FILENAME_IN_REPO = "names.db"


def publish(repo_id: str, db_path: str, token: str | None, api: HfApi | None = None) -> str:
    """Upload db_path to the given Hugging Face dataset repo, creating it if needed.

    api is injectable for testing; production callers leave it unset and get
    a real HfApi.
    """
    if not Path(db_path).exists():
        raise SystemExit(f"Database not found: {db_path}. Run `make build-db` first.")

    if not token:
        raise SystemExit(
            "HF_TOKEN is required to publish (even though the resulting dataset is public -- "
            "Hugging Face requires authentication to write, just not to read)."
        )

    api = api or HfApi(token=token)

    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=False,
        exist_ok=True,
    )

    return api.upload_file(
        path_or_fileobj=db_path,
        path_in_repo=FILENAME_IN_REPO,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Publish built database",
    )


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(f"Usage: {sys.argv[0]} <repo_id> [db_path]")
    repo_id = sys.argv[1]
    db_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_DB

    result = publish(repo_id, db_path, token=os.environ.get("HF_TOKEN"))
    print(f"Published {db_path} to {repo_id}: {result}")


if __name__ == "__main__":
    main()
