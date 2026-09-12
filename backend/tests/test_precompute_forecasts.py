"""Tests for the precompute batch itself.

The batch is otherwise exercised only indirectly, through the `sample_db`
session fixture in conftest.py. These cover the properties that fixture cannot
assert: that a published artifact is reproducible, that the extraction reads
the index rather than sorting, that one fit serves every name, and that the
coverage aggregate is derived from the stored table rather than from whatever
one invocation happened to fit.
"""

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.forecast import pooled  # noqa: E402
from scripts.make_sample_db import build  # noqa: E402
from scripts.precompute_forecasts import run  # noqa: E402


def _forecasts(db_path: str) -> dict:
    conn = sqlite3.connect(db_path)
    try:
        return {
            (name, sex): payload
            for name, sex, payload in conn.execute("SELECT name, sex, payload FROM forecasts")
        }
    finally:
        conn.close()


def _calibration(db_path: str) -> dict:
    conn = sqlite3.connect(db_path)
    try:
        return {
            (level, tier, bin_index): (coverage, n)
            for level, tier, bin_index, coverage, n in conn.execute(
                "SELECT nominal_level, tier, volatility_bin, empirical_coverage, n FROM calibration"
            )
        }
    finally:
        conn.close()


def _strata(db_path: str) -> dict:
    conn = sqlite3.connect(db_path)
    try:
        return {
            (name, sex): (tier, bin_index)
            for name, sex, tier, bin_index in conn.execute(
                "SELECT name, sex, tier, volatility_bin FROM forecasts"
            )
        }
    finally:
        conn.close()


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("precompute") / "names.db"
    build(str(path))
    return str(path)


def _copy(built: str, tmp_path, name: str) -> str:
    target = tmp_path / name
    target.write_bytes(Path(built).read_bytes())
    return str(target)


def test_two_runs_over_the_same_data_produce_identical_forecasts(built, tmp_path):
    """A published artifact has to be reproducible.

    Boosting subsamples rows and columns, and LightGBM builds its histograms
    in parallel, so nothing about the fit is reproducible by default — it is
    reproducible because the batch pins a seed and a thread count. Without
    both, rebuilding the same database twice would publish two different sets
    of numbers and there would be no way to tell a real model change from
    noise.

    Both runs go through the CLI, in separate processes. Calling `run` twice
    in one process would pass even with a seed drawn at import time, since
    both calls would draw the same one — it is exactly the run-to-run case
    that `make precompute-forecasts` is, and the only one worth pinning.
    """
    first = _copy(built, tmp_path, "first.db")
    second = _copy(built, tmp_path, "second.db")

    for db in (first, second):
        completed = subprocess.run(
            [sys.executable, "scripts/precompute_forecasts.py", db, "--threads", "2"],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr

    assert _forecasts(first) == _forecasts(second)
    assert _calibration(first) == _calibration(second)


def test_every_eligible_name_is_forecast_from_one_pooled_fit(built, tmp_path):
    """One model, every name — not one model per name.

    The count that matters is that the set of names stored is exactly the set
    the eligibility rule admits at the newest year, produced without ever
    looking at a name in isolation.
    """
    db = _copy(built, tmp_path, "pooled.db")
    result = run(db)

    conn = sqlite3.connect(db)
    try:
        (newest,) = conn.execute("SELECT MAX(year) FROM names").fetchone()
        eligible = {
            (name.lower(), sex)
            for name, sex in conn.execute(
                "SELECT name, sex FROM names GROUP BY LOWER(name), sex "
                "HAVING COUNT(*) >= 10 AND MAX(year) = ?",
                (newest,),
            )
        }
    finally:
        conn.close()

    assert eligible
    assert set(_forecasts(db)) == eligible
    assert result["stored"] == len(eligible)
    assert result["origins"]["production"] == newest


def test_extraction_reads_the_index_and_sorts_nothing(built):
    """The whole feature build is one indexed pass over `names`.

    `idx_names_name_sex_year` (ADR 0009) already stores rows in exactly the
    order the extraction wants them, so SQLite can hand them back grouped and
    ordered with no temporary B-tree and no sort file. Losing that — by
    reordering the query, or by dropping the index — turns a sub-second scan
    into a spill to disk on 11 million rows, which is the failure this pins.
    """
    conn = sqlite3.connect(built)
    try:
        plan = "\n".join(row[3] for row in conn.execute(f"EXPLAIN QUERY PLAN {pooled.SERIES_SQL}"))
    finally:
        conn.close()

    assert "idx_names_name_sex_year" in plan, plan
    assert "TEMP B-TREE" not in plan, plan


def test_forecasts_cover_only_years_that_have_not_happened_yet(built, tmp_path):
    """2025 is history, not a forecast point.

    The artifact this replaces carried forecasts for 2025-2029 produced before
    2025 was observed, so the search chart held two different values for 2025
    and the forecast overwrote the record. A forecast must start the year
    after the newest observation.
    """
    db = _copy(built, tmp_path, "years.db")
    run(db)

    conn = sqlite3.connect(db)
    try:
        (newest,) = conn.execute("SELECT MAX(year) FROM names").fetchone()
    finally:
        conn.close()

    stored = _forecasts(db)
    assert stored
    for payload in stored.values():
        years = [point["year"] for point in json.loads(payload)["forecast"]]
        assert years == list(range(newest + 1, newest + 6))


def test_no_forecast_or_band_edge_is_negative(built, tmp_path):
    """A share cannot be negative, and nothing clamps one any more.

    The forecast is the origin's share times the exponential of a predicted
    log growth, and each band edge is that times another exponential, so every
    published number is positive by construction. The old pipeline needed a
    `max(x, 0)` on every field because ARIMA's additive intervals could reach
    below zero; removing that clamp is only safe if this holds.
    """
    db = _copy(built, tmp_path, "positive.db")
    run(db)

    stored = _forecasts(db)
    assert stored
    for payload in stored.values():
        for point in json.loads(payload)["forecast"]:
            assert point["lo95"] <= point["lo80"] <= point["hi80"] <= point["hi95"]
            assert point["lo95"] > 0


def test_validation_scores_the_five_years_before_the_newest_one(built, tmp_path):
    """The holdout has to be fully observed, or it is not a holdout.

    Training at five years back and scoring against what actually happened is
    what makes the search page's predicted-against-actual table checkable
    against recorded history.
    """
    db = _copy(built, tmp_path, "validation.db")
    result = run(db)

    conn = sqlite3.connect(db)
    try:
        (newest,) = conn.execute("SELECT MAX(year) FROM names").fetchone()
    finally:
        conn.close()

    assert result["origins"]["holdout"] == newest - 5
    assert result["validated"] > 0

    validations = [json.loads(payload)["validation"] for payload in _forecasts(db).values()]
    scored = [v for v in validations if v is not None]
    assert scored
    for validation in scored:
        assert [point["year"] for point in validation["points"]] == list(
            range(newest - 4, newest + 1)
        )


def test_coverage_is_stored_per_name_and_never_served(built, tmp_path):
    """`payload` is handed to the API verbatim, so coverage must not be in it."""
    db = _copy(built, tmp_path, "coverage.db")
    run(db)

    conn = sqlite3.connect(db)
    try:
        rows = conn.execute(
            "SELECT payload, coverage_hits, coverage_n FROM forecasts WHERE coverage_n IS NOT NULL"
        ).fetchall()
    finally:
        conn.close()

    assert rows, "expected at least one name with a holdout backtest"
    for payload, hits, counts in rows:
        validation = json.loads(payload)["validation"]
        if validation is not None:
            assert "coverage" not in validation
        assert json.loads(hits).keys() == json.loads(counts).keys()


def test_it_replaces_a_forecasts_table_left_by_the_previous_pipeline(built, tmp_path):
    """An existing artifact must not have to be rebuilt from scratch.

    `data/names.built.db` carries a `forecasts` table from the ARIMA batch —
    three columns wide, and holding forecasts for years that have since been
    observed. `CREATE TABLE IF NOT EXISTS` will not widen it, and leaving its
    rows in place would serve stale years alongside the new ones.
    """
    db = _copy(built, tmp_path, "legacy.db")

    conn = sqlite3.connect(db)
    conn.execute("DROP TABLE IF EXISTS forecasts")
    conn.execute(
        "CREATE TABLE forecasts ("
        "name TEXT NOT NULL, sex TEXT NOT NULL, payload TEXT NOT NULL, "
        "PRIMARY KEY (name, sex))"
    )
    conn.execute(
        "INSERT INTO forecasts (name, sex, payload) VALUES ('zzz-retired', 'F', '{}')",
    )
    conn.commit()
    conn.close()

    result = run(db)

    assert result["stored"] > 0
    assert ("zzz-retired", "F") not in _forecasts(db)
    assert _calibration(db)


def test_the_model_card_is_stored_once_rather_than_per_name(built, tmp_path):
    """One pooled model means one description of it.

    Copying the card into every payload would repeat the same few hundred
    bytes ~24,700 times in the published artifact and leave room for two rows
    to disagree about what produced them.
    """
    db = _copy(built, tmp_path, "card.db")
    run(db)

    conn = sqlite3.connect(db)
    try:
        cards = conn.execute("SELECT payload FROM model_card").fetchall()
    finally:
        conn.close()

    assert len(cards) == 1
    card = json.loads(cards[0][0])
    assert card["features"] == list(pooled.FEATURES)
    assert card["training_rows"] > 0
    for payload in _forecasts(db).values():
        assert "model" not in json.loads(payload)


def test_the_published_forecasts_add_up_to_the_share_each_sex_held(built, tmp_path):
    """Reconciliation reaches the artifact, not just the module that does it.

    Shares within a sex sum to a fixed total, so the forecasts have to as
    well. The batch's own point stack smooths each path and then scales each
    (sex, horizon) slice onto the total observed at the origin; if either step
    were dropped on the way into `forecasts`, this is where it would show.
    """
    db = _copy(built, tmp_path, "adds-up.db")
    run(db)

    conn = sqlite3.connect(db)
    try:
        (newest,) = conn.execute("SELECT MAX(year) FROM names").fetchone()
        origin_share = {
            (name.lower(), sex): percent
            for name, sex, percent in conn.execute(
                "SELECT name, sex, popularity_percent FROM names WHERE year = ?", (newest,)
            )
        }
    finally:
        conn.close()

    totals: dict[str, list[float]] = {}
    targets: dict[str, float] = {}
    for (name, sex), payload in _forecasts(db).items():
        points = json.loads(payload)["forecast"]
        # The constraint is checked over the years actually published — the
        # five after the origin, which on the 2025 database is 2026-2030.
        assert [point["year"] for point in points] == list(range(newest + 1, newest + 6))
        means = [point["mean"] for point in points]
        totals[sex] = [a + b for a, b in zip(totals.get(sex, [0.0] * 5), means, strict=True)]
        targets[sex] = targets.get(sex, 0.0) + origin_share[(name, sex)]

    assert totals
    for sex, forecast_total in totals.items():
        for value in forecast_total:
            assert value == pytest.approx(targets[sex], rel=1e-9)


def test_the_model_card_reports_the_bounded_training_window(built, tmp_path):
    """The card describes the fit that happened, window included."""
    db = _copy(built, tmp_path, "window.db")
    result = run(db)

    conn = sqlite3.connect(db)
    try:
        (payload,) = conn.execute("SELECT payload FROM model_card WHERE id = 1").fetchone()
        (newest,) = conn.execute("SELECT MAX(year) FROM names").fetchone()
    finally:
        conn.close()

    card = json.loads(payload)
    assert card == result["model"]
    assert card["training_origins"] == len(pooled.training_origins(newest))
    assert card["training_origins"] <= pooled.TRAIN_WINDOW


def test_calibration_is_keyed_by_level_tier_and_volatility_bin(built, tmp_path):
    """Coverage is reported per stratum, because that is where it goes wrong.

    A single population figure can read 0.80 while the steady names are
    covered 0.95 of the time and the jumpy ones 0.60 — and it is the jumpy
    name's visitor who is being told 80%. One row per
    `(level, tier, volatility bin)` is what lets the chart label a band with
    the coverage measured for names like the one being looked at.
    """
    db = _copy(built, tmp_path, "strata.db")
    run(db)

    calibration = _calibration(db)

    assert calibration
    levels = {level for level, _, _ in calibration}
    assert levels == {0.8, 0.95}
    for level in levels:
        strata = {(tier, b) for lv, tier, b in calibration if lv == level}
        assert strata
        for tier, bin_index in strata - {pooled.GLOBAL_STRATUM}:
            assert tier in pooled.TIERS
            assert 0 <= bin_index < pooled.VOLATILITY_BINS
    for coverage, n in calibration.values():
        assert 0.0 <= coverage <= 1.0
        assert n > 0


def test_the_population_row_holds_only_the_names_that_were_given_that_band(built, tmp_path):
    """The fallback row reports the fallback band, not everything pooled.

    A name whose stratum was too thin to earn a band was given the global one,
    and the global row is what it will be reported. So that row has to be
    measured on those names — pooling in the names that got a *different*,
    narrower or wider band of their own would make it describe a band nobody
    holds.
    """
    db = _copy(built, tmp_path, "population.db")
    result = run(db)

    calibration = _calibration(db)
    global_tier, global_bin = pooled.GLOBAL_STRATUM

    for level in (0.8, 0.95):
        cells = {(tier, b): value for (lv, tier, b), value in calibration.items() if lv == level}
        assert cells
        total = sum(n for _, n in cells.values())
        # The headline the CLI prints pools every cell; it is deliberately not
        # a row in the table, because it describes no single band.
        pooled_hits = sum(coverage * n for coverage, n in cells.values())
        assert result["calibration"][str(level)] == pytest.approx(pooled_hits / total, rel=1e-9)
        if (global_tier, global_bin) in cells:
            assert cells[(global_tier, global_bin)][1] < total or len(cells) == 1


def test_each_forecast_carries_the_stratum_it_is_served_under(built, tmp_path):
    """Serving tiers a name by its rank in the newest observed year.

    The band a visitor sees is the one for this name's stratum *now*, so the
    tier stored beside the forecast has to be read from the newest year's
    rank — not from the historical origin the bands were calibrated at, and
    not from wherever the name's series happens to end.
    """
    db = _copy(built, tmp_path, "served.db")
    run(db)

    conn = sqlite3.connect(db)
    try:
        (newest,) = conn.execute("SELECT MAX(year) FROM names").fetchone()
        newest_rank = {
            (name.lower(), sex): rank
            for name, sex, rank in conn.execute(
                "SELECT name, sex, popularity_rank FROM names WHERE year = ?", (newest,)
            )
        }
    finally:
        conn.close()

    strata = _strata(db)

    assert strata
    for key, (tier, bin_index) in strata.items():
        assert tier == pooled.popularity_tier(newest_rank[key])
        assert 0 <= bin_index < pooled.VOLATILITY_BINS


def test_only_names_with_a_scored_holdout_contribute_to_calibration(built, tmp_path):
    """Coverage rests on holdouts that happened, so an absent name adds nothing.

    The sample database deliberately carries a name that has fallen out of use
    and one with too little history, neither of which is forecast at all — and
    an eligible name whose holdout window is incomplete has a forecast but no
    scored points. The published `n` has to be exactly the points that were
    actually checked, or the coverage fraction is diluted by names that were
    never tested.
    """
    db = _copy(built, tmp_path, "ineligible.db")
    run(db)

    conn = sqlite3.connect(db)
    try:
        all_names = {
            (name.lower(), sex)
            for name, sex in conn.execute("SELECT DISTINCT name, sex FROM names")
        }
    finally:
        conn.close()

    stored = _forecasts(db)
    scored = [
        payload for payload in stored.values() if json.loads(payload)["validation"] is not None
    ]
    calibration = _calibration(db)

    assert all_names - set(stored), "expected the sample to hold an unforecastable name"
    assert scored
    for level in (0.8, 0.95):
        counted = sum(n for (lv, _, _), (_, n) in calibration.items() if lv == level)
        assert counted == len(scored) * pooled.H


def test_a_corpus_big_enough_gets_bands_of_its_own_per_stratum(stratified_db):
    """Conditioning has to actually happen once there are rows to condition on.

    On the sample database every cell is below `MIN_STRATUM_ROWS` and every
    band correctly falls back to the population's, so nothing there can tell a
    conditioned pipeline from an unconditioned one. This corpus fills the
    cells. See docs/adr/0011-conformal-bands-keyed-by-strata.md.
    """
    calibration = _calibration(stratified_db)

    assert calibration
    for level in (0.8, 0.95):
        strata = {(tier, b) for lv, tier, b in calibration if lv == level}
        assert len(strata - {pooled.GLOBAL_STRATUM}) > 1, strata
        for tier, bin_index in strata - {pooled.GLOBAL_STRATUM}:
            assert tier in pooled.TIERS
            assert 0 <= bin_index < pooled.VOLATILITY_BINS
        for (lv, tier, b), (_coverage, n) in calibration.items():
            if lv == level and (tier, b) != pooled.GLOBAL_STRATUM:
                assert n >= pooled.MIN_STRATUM_ROWS


def test_a_volatile_name_is_published_with_a_wider_band(stratified_db):
    """End to end: the width a visitor sees depends on this name's own wobble.

    Two names at the same popularity tier, one in the steadiest volatility bin
    and one in the jumpiest, must reach `forecasts` with visibly different
    band widths — or the conditioning is happening somewhere that the stored
    artifact never sees.
    """
    strata = _strata(stratified_db)
    stored = _forecasts(stratified_db)

    def width(key) -> float:
        point = json.loads(stored[key])["forecast"][-1]
        return point["hi95"] / point["lo95"]

    tier = "top100"
    steady = [key for key, (t, b) in strata.items() if t == tier and b == 0]
    lurching = [key for key, (t, b) in strata.items() if t == tier and b == 2]

    assert steady and lurching
    narrowest = min(width(key) for key in lurching)
    widest = max(width(key) for key in steady)
    assert narrowest > widest, (widest, narrowest)


def _model_evaluation(db_path: str) -> dict:
    conn = sqlite3.connect(db_path)
    try:
        return {
            tier: {
                "pool_skill": pool_skill,
                "med_skill": med_skill,
                "origins_evaluated": origins,
                "min_origin": min_origin,
                "max_origin": max_origin,
            }
            for tier, pool_skill, med_skill, origins, min_origin, max_origin in conn.execute(
                "SELECT tier, pool_skill, med_skill, origins_evaluated, min_origin, max_origin "
                "FROM model_evaluation"
            )
        }
    finally:
        conn.close()


def test_skill_is_measured_across_every_window_since_1995(built, tmp_path):
    """The figure on the chart is a property of the name, not of one window.

    Scored on the 2021-25 holdout alone, every name is labelled with how the
    birth-rate shock went for it, and next year's rebuild relabels it with
    something else. The batch scores each name at every origin it was eligible
    at since 1995 and stores the average, so a long-lived name's figure rests
    on 26 measurements rather than one.
    """
    db = _copy(built, tmp_path, "span.db")
    result = run(db)

    conn = sqlite3.connect(db)
    try:
        (newest,) = conn.execute("SELECT MAX(year) FROM names").fetchone()
    finally:
        conn.close()

    span = list(range(1995, newest - 4))
    assert result["backtest"]["origins"] == span
    assert len(span) == 26

    scored = [
        json.loads(payload)["validation"]
        for payload in _forecasts(db).values()
        if json.loads(payload)["validation"] is not None
    ]
    assert scored
    windows = [validation["skill_windows"] for validation in scored]
    assert max(windows) == len(span)
    assert all(1 <= count <= len(span) for count in windows)
    # A name recorded for less of the span is averaged over less of it rather
    # than dropped: the counts differ, and none of them is zero.
    assert len(set(windows)) > 1


def test_the_artifact_carries_the_scores_it_would_be_deployed_on(built, tmp_path):
    """`model_evaluation` is how a built database certifies itself.

    The deploy gate has no access to the run that produced an artifact — only
    to the artifact. So the measured tier scores travel inside it, beside the
    forecasts they describe, with the span they were measured over attached so
    a truncated backtest cannot pass itself off as a full one.
    """
    db = _copy(built, tmp_path, "evaluation.db")
    result = run(db)

    evaluation = _model_evaluation(db)

    assert evaluation == result["backtest"]["evaluation"]
    assert set(evaluation) <= set(pooled.TIERS)
    assert evaluation
    for scores in evaluation.values():
        assert -1.0 <= scores["pool_skill"] <= 1.0
        assert -1.0 <= scores["med_skill"] <= 1.0
        assert scores["origins_evaluated"] == 26
        assert scores["min_origin"] == 1995
        assert scores["max_origin"] == result["origins"]["holdout"]


def test_a_database_too_early_to_backtest_still_builds_an_artifact(tmp_path):
    """No window has closed since 1995, so nothing is scored — and it says so.

    `pooled.backtest_span` reaches back to 1995, so a database whose newest
    year is earlier than 2000 offers not one five-year window that has closed
    since. That is a development corpus rather than a deployable one, and the
    batch's job is to say which: it fits, forecasts and calibrates as usual,
    the holdout figures are stored because they are true, and `skill` is
    absent rather than a zero nobody measured. `model_evaluation` is empty for
    the same reason, which is what makes `verify_db` refuse to ship it.
    """
    path = tmp_path / "early.db"
    _write_corpus(str(path), last_year=1999)

    result = run(str(path))

    assert result["backtest"]["origins"] == []
    assert result["backtest"]["evaluation"] == {}
    stored = _forecasts(str(path))
    assert stored
    validations = [json.loads(payload)["validation"] for payload in stored.values()]
    scored = [validation for validation in validations if validation is not None]
    assert scored, "the holdout window is fully observed, so it is still scored"
    for validation in scored:
        assert "mae" in validation
        assert "skill" not in validation
        assert "skill_windows" not in validation


def test_the_deploy_gate_refuses_an_artifact_nothing_was_scored_on(tmp_path):
    """The build above is exactly the one that must not reach production."""
    from scripts.verify_db import VerificationError, verify

    path = tmp_path / "early.db"
    _write_corpus(str(path), last_year=1999)
    run(str(path))

    with pytest.raises(VerificationError, match="model_evaluation"):
        verify(str(path))


def _write_corpus(path: str, last_year: int) -> None:
    """A small corpus of plain wobbling series, ending where the caller says."""
    import math

    from app import db_schema

    rows = []
    for i in range(30):
        name, sex = f"name{i:02d}", "F" if i % 2 else "M"
        for year in range(last_year - 39, last_year + 1):
            value = 0.001 * (1.0 + 0.1 * math.sin(i + year / 5))
            rows.append((name, sex, int(value * 1_000_000), year, value, (i % 20) + 1))

    conn = sqlite3.connect(path)
    conn.execute(db_schema.CREATE_TABLE)
    conn.executemany("INSERT INTO names VALUES (?, ?, ?, ?, ?, ?)", rows)
    db_schema.create_indexes(conn)
    conn.commit()
    conn.close()
