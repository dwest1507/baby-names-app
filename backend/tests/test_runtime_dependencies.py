"""The application package must run without the fitting libraries installed.

The production image builds with `uv sync --frozen --no-dev`, so anything in
the dev/batch dependency group is simply absent from the container. Asserting
that the app *doesn't import* statsmodels or scipy is weaker than asserting it
*runs without them*, so these tests block the imports outright and import the
app for real. See docs/adr/0004-forecasts-as-a-build-artifact.md.
"""

import subprocess
import sys
import tomllib
from pathlib import Path

BACKEND = Path(__file__).parent.parent

# Refuses statsmodels, scipy and their submodules the way a container that
# never installed them would, then exercises the request path's module graph.
BLOCKED = """
import sys

BANNED = ("statsmodels", "scipy")


class Blocker:
    def find_module(self, name, path=None):
        return self.find_spec(name, path)

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BANNED:
            raise ImportError(f"{name} is not installed in the runtime image")
        return None


sys.meta_path.insert(0, Blocker())

import app.main  # noqa: F401
from app.services import forecast

assert forecast.is_eligible([2016, 2025], 2025) is False
composed = forecast.build_response(
    "F",
    [{"name": "emma", "year": 2025, "popularity_percent": 0.01}],
    None,
)
assert composed["forecast"] == []
print("ok")
"""


def _run_without_fitting_libraries() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", BLOCKED],
        cwd=BACKEND,
        capture_output=True,
        text=True,
    )


def test_the_app_imports_without_statsmodels_or_scipy():
    result = _run_without_fitting_libraries()
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_the_fitting_libraries_are_not_runtime_dependencies():
    # The image builds with `uv sync --frozen --no-dev`, which installs exactly
    # [project.dependencies]. Anything absent from that list is absent from the
    # container, so this is the list the container footprint is decided by.
    manifest = tomllib.loads((BACKEND / "pyproject.toml").read_text())
    runtime = {
        requirement.split(">")[0].split("=")[0].split("[")[0].strip()
        for requirement in manifest["project"]["dependencies"]
    }
    assert "statsmodels" not in runtime
    assert "scipy" not in runtime
