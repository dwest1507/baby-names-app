import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# The chatbot degrades gracefully when no key is configured, so this is optional
GROQ_API_KEY: str | None = os.environ.get("GROQ_API_KEY")
GROQ_MODEL: str = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

ALLOWED_ORIGINS: list[str] = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

REPO_ROOT = Path(__file__).parent.parent.parent
DEFAULT_DB_PATH = str(REPO_ROOT / "data" / "names.db")

NAMES_DB_PATH: str = os.environ.get("NAMES_DB_PATH", DEFAULT_DB_PATH)
NAMES_DB_REPO: str | None = os.environ.get("NAMES_DB_REPO")
NAMES_DB_FILE: str = os.environ.get("NAMES_DB_FILE", "names.db")
NAMES_DB_REPO_TYPE: str = os.environ.get("NAMES_DB_REPO_TYPE", "dataset")
HF_TOKEN: str | None = os.environ.get("HF_TOKEN")
