"""The ported pooled pipeline must reproduce the research harness's numbers.

`scripts/forecast/pooled.py` is a port: the feature block, the target, the
weights and the hyperparameters were all settled in `research/forecasting/`
over six rounds of rolling-origin benchmarks, and the code that shipped is a
tidied copy rather than the code that measured. A port can be subtly wrong in
ways no shape assertion notices — a feature computed over eleven points
instead of ten, a weight normalised after clipping instead of before — and the
result is a model that is merely *similar* to the one whose skill was
published.

So the numbers are pinned. `research/forecasting/make_parity_fixture.py`
generates a set of series, runs them through the **research** modules, and
writes both the series and the predictions those modules produced into
`tests/fixtures/pooled_parity.json`. This test replays the same series through
the shipped code and demands the same answers.

The fixture is what lets that happen in CI: it is a few hundred kilobytes of
checked-in JSON, so the check needs neither the 1.1 GB source database nor the
research harness. Nothing here imports `research/` — if it did, the harness
would be a test dependency of the backend, which is the coupling the fixture
exists to avoid.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.forecast import pooled  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "pooled_parity.json"


@pytest.fixture(scope="module")
def parity() -> dict:
    return json.loads(FIXTURE.read_text())


@pytest.fixture(scope="module")
def series(parity) -> list:
    return [
        (
            entry["key"],
            np.array(entry["years"], dtype=np.int32),
            np.array(entry["values"], dtype=np.float64),
            entry["rank"],
        )
        for entry in parity["series"]
    ]


def test_the_fixture_is_small_enough_to_live_in_git():
    """A fixture nobody wants to clone is a fixture that gets deleted."""
    assert FIXTURE.stat().st_size < 450_000


def test_the_feature_block_matches_the_researched_one(parity, series):
    """Every feature of every row, against what the research code computed.

    This is the half of the port that a prediction comparison could mask: a
    booster trained on a slightly wrong column can still land close enough to
    look right on average.
    """
    rows = pooled.build_rows(series, [parity["origin"]])
    expected = parity["features"]

    assert [row["key"] for row in rows] == [entry["key"] for entry in expected]
    assert list(pooled.FEATURES) == parity["feature_names"]
    for row, entry in zip(rows, expected, strict=True):
        np.testing.assert_allclose(row["x"], entry["x"], rtol=1e-12, atol=1e-12)


def test_the_shipped_pipeline_reproduces_the_researched_forecasts(parity, series):
    """The whole pipeline, end to end, against the pinned predictions."""
    training = pooled.training_rows(series, parity["origin"])
    assert len(training) == parity["training_rows"]
    assert sorted({row["origin"] for row in training}) == parity["training_origins"]

    models = pooled.train(training, seed=parity["seed"], threads=parity["threads"])
    rows = pooled.build_rows(series, [parity["origin"]])
    predicted = pooled.predict(models, rows)

    # Nine significant figures rather than bit equality: the shipped fit adds
    # LightGBM's `deterministic` flag, which the harness does not set, and that
    # changes the last couple of bits of a leaf value. The flag is the reason
    # the batch is reproducible run to run, so it stays and the tolerance
    # absorbs it — a real porting error moves these numbers far further than
    # this, because it moves a split rather than a rounding.
    np.testing.assert_allclose(predicted, parity["predicted"], rtol=1e-9, atol=1e-15)


def test_the_shipped_point_stack_reproduces_the_researched_forecasts(parity, series):
    """And the three corrections that turn a prediction into the published line.

    The booster's output is not what the site draws. Research capped each
    path's implied growth, smoothed it, and scaled each (sex, horizon) slice
    onto the origin's total, in that order; `pooled.point_forecasts` is the
    port of those three, and this pins it against the harness modules that
    measured them (`cap.py`, `smooth.py`, `reconcile.py`).
    """
    training = pooled.training_rows(series, parity["origin"])
    models = pooled.train(training, seed=parity["seed"], threads=parity["threads"])
    rows = pooled.build_rows(series, [parity["origin"]])

    assert pooled.TRAIN_WINDOW == parity["window"]
    assert pooled.CAP_QUANTILE == parity["cap_quantile"]

    # Pinned on its own because it does not show up in the forecasts: a bound
    # at the 99.9th percentile of what names actually do is one no ordinary
    # forecast reaches, so nothing in this fixture is clipped and a cap
    # derived from the wrong rows would leave the published numbers identical.
    caps = pooled.growth_caps(training)
    np.testing.assert_allclose(caps, parity["caps"], rtol=1e-12)

    published = pooled.point_forecasts(models, rows, caps)
    np.testing.assert_allclose(published, parity["published"], rtol=1e-9, atol=1e-15)


# Refits the fixture in a fresh interpreter and prints the forecasts, so the
# comparison below spans two processes rather than two calls in one.
REFIT = """
import json, sys
sys.path.insert(0, %r)
import numpy as np
from scripts.forecast import pooled

parity = json.loads(open(%r).read())
series = [
    (e["key"], np.array(e["years"], dtype=np.int32), np.array(e["values"]), e["rank"])
    for e in parity["series"]
]
models = pooled.train(pooled.training_rows(series, parity["origin"]), threads=parity["threads"])
rows = pooled.build_rows(series, [parity["origin"]])
print(json.dumps(pooled.predict(models, rows).tolist()))
"""


def test_the_pipeline_is_bit_identical_across_processes(parity, series):
    """Rebuilding the artifact must not quietly republish different numbers.

    This is the fixture where determinism means something: the sample database
    is too small for LightGBM to make a single split, so two runs there agree
    about a constant. Here there is a real ensemble, and it has to come out the
    same in a fresh interpreter — which is what `make precompute-forecasts`
    always is.
    """
    training = pooled.training_rows(series, parity["origin"])
    rows = pooled.build_rows(series, [parity["origin"]])
    here = pooled.predict(pooled.train(training, threads=parity["threads"]), rows)

    backend = Path(__file__).parent.parent
    completed = subprocess.run(
        [sys.executable, "-c", REFIT % (str(backend), str(FIXTURE))],
        cwd=backend,
        capture_output=True,
        text=True,
        env={**os.environ, "FORECAST_THREADS": str(parity["threads"])},
    )
    assert completed.returncode == 0, completed.stderr

    assert np.array_equal(here, np.array(json.loads(completed.stdout)))


def test_the_model_the_fixture_pins_actually_splits(parity, series):
    """Guard against the fixture quietly degenerating into a constant.

    If the training set were too small or too uniform for LightGBM to make a
    split, every test above would still pass while pinning nothing but a
    weighted mean. A fixture that stops discriminating between names has
    stopped testing the model.
    """
    training = pooled.training_rows(series, parity["origin"])
    models = pooled.train(training, threads=parity["threads"])

    leaves = [model.booster_.trees_to_dataframe().shape[0] for model in models]
    assert min(leaves) > 100, leaves


def test_nothing_the_backend_runs_reaches_into_the_research_harness():
    """The harness is a benchmark, not a dependency of the shipped code.

    The dependency runs the other way round: `research/forecasting` imports the
    batch so that it benchmarks what actually ships. A backend module importing
    back would make the harness a build dependency and close that loop.
    """
    assert not [name for name in sys.modules if name.split(".")[0] == "research"]

    backend = Path(__file__).parent.parent
    for source in (*(backend / "app").rglob("*.py"), *(backend / "scripts").rglob("*.py")):
        text = source.read_text()
        assert "import research" not in text, source
        assert "from research" not in text, source
