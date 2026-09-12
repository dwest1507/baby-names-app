import re
import sqlite3
from pathlib import Path

import pytest
from conftest import SHARED_SECRET
from fastapi.testclient import TestClient

from app.main import app

# The frontend proxy is the only caller; it always attaches the shared secret.
client = TestClient(app, headers={"X-Backend-Secret": SHARED_SECRET})
# Anything else on the public internet looks like this.
anonymous = TestClient(app)


def test_endpoint_rejects_a_request_without_the_shared_secret():
    response = anonymous.get("/api/meta")
    assert response.status_code == 401


def _api_routes() -> list[tuple[str, str]]:
    """Every route the app serves, read from the app rather than hand-listed, so
    that a route added later is covered without editing this test."""
    routes = []
    for path, operations in app.openapi()["paths"].items():
        for method in operations:
            routes.append((method.upper(), re.sub(r"{[^}]+}", "emma", path)))
    return sorted(routes)


@pytest.mark.parametrize(("method", "path"), _api_routes())
def test_every_endpoint_but_health_rejects_a_request_without_the_secret(method, path):
    response = anonymous.request(method, path, params={"sex": "F"}, json={"message": "hi"})
    if path == "/api/health":
        assert response.status_code == 200
    else:
        assert response.status_code == 401


def test_health_answers_without_the_shared_secret():
    # The deployment platform probes this endpoint and cannot present a secret.
    response = anonymous.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["database"] == "ok"


def test_meta():
    response = client.get("/api/meta")
    assert response.status_code == 200
    assert response.json() == {"min_year": 1960, "max_year": 2025}


def test_top_names():
    response = client.get("/api/top-names", params={"sex": "F", "year": 2015, "limit": 5})
    assert response.status_code == 200
    names = response.json()["names"]
    assert 0 < len(names) <= 5
    assert names[0]["popularity_rank"] == 1


def test_top_names_validates_sex():
    response = client.get("/api/top-names", params={"sex": "X", "year": 2015})
    assert response.status_code == 422


def test_name_history():
    response = client.get("/api/names/emma", params={"sex": "F"})
    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "Emma"
    assert len(body["history"]) > 10


def test_name_history_not_found():
    response = client.get("/api/names/zzyzx", params={"sex": "M"})
    assert response.status_code == 404


def test_forecast():
    response = client.get("/api/names/emma/forecast", params={"sex": "F"})
    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "Emma"
    assert len(body["history"]) >= 10
    assert len(body["forecast"]) == 5
    first = body["forecast"][0]
    assert first["year"] == body["history"][-1]["year"] + 1
    assert first["lo95"] <= first["mean"] <= first["hi95"]
    assert body["model"] is not None
    assert body["validation"] is not None


def test_forecast_validation_carries_skill_against_naive_persistence():
    # Skill compares the model's holdout MAE against a naive baseline that
    # simply repeats the last training-observed value. See
    # docs/adr/0011-conformal-bands-keyed-by-strata.md.
    response = client.get("/api/names/emma/forecast", params={"sex": "F"})
    assert response.status_code == 200
    skill = response.json()["validation"]["skill"]
    assert isinstance(skill, float)
    assert skill <= 1.0


def test_forecast_carries_measured_interval_calibration():
    # The published bands must be labelled with the coverage they actually
    # achieve, measured across every eligible name's holdout backtest — not
    # the nominal level. See
    # docs/adr/0011-conformal-bands-keyed-by-strata.md.
    response = client.get("/api/names/emma/forecast", params={"sex": "F"})
    assert response.status_code == 200
    calibration = response.json()["calibration"]
    assert set(calibration) == {"0.8", "0.95"}
    for level, key in ((0.8, "0.8"), (0.95, "0.95")):
        entry = calibration[key]
        assert entry["nominal"] == level
        assert 0.0 <= entry["empirical_coverage"] <= 1.0
        assert entry["n"] > 5


def test_forecast_calibration_names_the_stratum_it_describes():
    """The figure served says which names it was measured over.

    Without that the response is a bare percentage again, and a
    whole-population fallback is indistinguishable from a measurement of this
    name's own cell.
    """
    response = client.get("/api/names/emma/forecast", params={"sex": "F"})
    assert response.status_code == 200
    calibration = response.json()["calibration"]

    for entry in calibration.values():
        assert entry["tier"] in ("top100", "top1000", "top5000", "rest", "*")
        assert entry["volatility_bin"] >= -1


def test_forecast_calibration_describes_this_names_own_stratum(use_stratified_db):
    """Two names in different strata are given different coverage rows.

    This is the whole point of keying `calibration` by stratum: handing every
    name one aggregate is what let a good population average conceal a badly
    calibrated tail. The sample database is too small for any cell to earn a
    band of its own, so this runs against a corpus where they do — see the
    `stratified_db` fixture.
    """
    rows = {}
    for name in ("name0000", "name0300", "name0629", "name0001", "name0301"):
        sex = "F" if int(name[-4:]) % 2 else "M"
        body = client.get(f"/api/names/{name}/forecast", params={"sex": sex}).json()
        assert body["forecast"], name
        entry = body["calibration"]["0.8"]
        rows[name] = (entry["tier"], entry["volatility_bin"], entry["empirical_coverage"])

    strata = {(tier, b) for tier, b, _ in rows.values()}
    assert len(strata) > 1, rows
    assert ("*", -1) not in strata, rows
    assert len({coverage for _, _, coverage in rows.values()}) > 1, rows


def test_forecast_carries_the_names_own_stratum():
    """The page labels the name with its tier and volatility bin.

    Not with the stratum of the band it was handed: a name whose own cell was
    too thin to earn a band is served the population's, and `calibration`
    then says `*`/-1. Reporting that as the name's popularity tier would be
    false. The two are different facts and the response carries both.
    """
    response = client.get("/api/names/emma/forecast", params={"sex": "F"})
    assert response.status_code == 200
    stratum = response.json()["stratum"]
    assert stratum["tier"] in ("top100", "top1000", "top5000", "rest")
    assert stratum["volatility_bin"] in (0, 1, 2)


def test_forecast_stratum_is_absent_where_there_is_no_forecast():
    response = client.get("/api/names/debra/forecast", params={"sex": "F"})
    assert response.status_code == 200
    body = response.json()
    assert body["forecast"] == []
    assert body["stratum"] is None


def test_chat_unavailable_without_key(monkeypatch):
    from app import config

    monkeypatch.setattr(config, "GROQ_API_KEY", None)
    response = client.post("/api/chat", json={"message": "How many names?"})
    assert response.status_code == 503


def test_name_history_omits_years_with_no_recorded_births():
    # Debra falls out of use partway through the sample data; the fabricated
    # zero rows that pad it to the final year must not appear as history.
    newest_year = client.get("/api/meta").json()["max_year"]
    response = client.get("/api/names/debra", params={"sex": "F"})
    assert response.status_code == 200
    history = response.json()["history"]
    assert history
    assert all(row["total_count"] > 0 for row in history)
    assert history[-1]["year"] < newest_year


def test_forecast_omitted_for_a_name_no_longer_in_use():
    # Debra has ample history but was not recorded in the newest year, so it is
    # not in current use and must not be forecast.
    response = client.get("/api/names/debra/forecast", params={"sex": "F"})
    assert response.status_code == 200
    body = response.json()
    assert len(body["history"]) >= 10
    assert body["forecast"] == []


def test_forecast_omitted_for_a_name_with_too_little_history():
    # Mateo is a recent arrival: recorded in the newest year, but with fewer
    # than MIN_HISTORY_YEARS observed years to fit on.
    response = client.get("/api/names/mateo/forecast", params={"sex": "M"})
    assert response.status_code == 200
    body = response.json()
    assert 0 < len(body["history"]) < 10
    assert body["forecast"] == []


def test_forecast_never_covers_a_year_that_has_already_occurred():
    newest_year = client.get("/api/meta").json()["max_year"]
    for name, sex in (("emma", "F"), ("debra", "F"), ("mateo", "M")):
        body = client.get(f"/api/names/{name}/forecast", params={"sex": sex}).json()
        assert all(point["year"] > newest_year for point in body["forecast"])


def test_forecast_for_a_name_in_current_use_covers_the_next_five_years():
    newest_year = client.get("/api/meta").json()["max_year"]
    body = client.get("/api/names/emma/forecast", params={"sex": "F"}).json()
    assert len(body["history"]) >= 10
    assert [point["year"] for point in body["forecast"]] == list(
        range(newest_year + 1, newest_year + 6)
    )


def test_forecast_endpoint_fits_no_model_at_request_time(monkeypatch):
    # Forecasts are precomputed by scripts/precompute_forecasts.py and stored;
    # the endpoint is a lookup. If it fit or even predicted live, this would
    # raise and the request would 500 instead of returning a populated
    # forecast. See docs/adr/0004-forecasts-as-a-build-artifact.md.
    from scripts.forecast import pooled

    def _must_not_be_called(*args, **kwargs):
        raise AssertionError("the pooled model must not run on the request path")

    monkeypatch.setattr(pooled, "train", _must_not_be_called)
    monkeypatch.setattr(pooled, "predict", _must_not_be_called)
    monkeypatch.setattr(pooled, "build_rows", _must_not_be_called)

    response = client.get("/api/names/emma/forecast", params={"sex": "F"})
    assert response.status_code == 200
    body = response.json()
    assert len(body["forecast"]) == 5
    assert body["model"] is not None


# One chat turn spends two provider calls, so the chat tier is much tighter than
# the general one; see app/limiter.py for the arithmetic.
CHAT_BURST = 5


def _no_provider_calls(monkeypatch):
    """Chat returns 503 without a key, so these tests never reach Groq."""
    from app import config

    monkeypatch.setattr(config, "GROQ_API_KEY", None)


def test_chat_is_limited_more_tightly_than_the_rest_of_the_api(monkeypatch):
    _no_provider_calls(monkeypatch)
    visitor = {"X-Forwarded-For": "203.0.113.10"}

    chat = [
        client.post("/api/chat", json={"message": "hi"}, headers=visitor).status_code
        for _ in range(CHAT_BURST + 1)
    ]
    assert chat[-1] == 429
    assert 429 not in chat[:-1]

    # The same number of ordinary requests is nowhere near the general ceiling.
    other = [client.get("/api/meta", headers=visitor).status_code for _ in range(CHAT_BURST + 1)]
    assert 429 not in other


def test_two_visitors_at_different_addresses_get_independent_buckets(monkeypatch):
    _no_provider_calls(monkeypatch)
    one = {"X-Forwarded-For": "203.0.113.10"}
    another = {"X-Forwarded-For": "198.51.100.7"}

    for _ in range(CHAT_BURST):
        client.post("/api/chat", json={"message": "hi"}, headers=one)
    assert client.post("/api/chat", json={"message": "hi"}, headers=one).status_code == 429

    # A different visitor arriving through the same proxy is unaffected.
    assert client.post("/api/chat", json={"message": "hi"}, headers=another).status_code == 503


# The general ceiling: 60 requests a minute per visitor, across all endpoints.
GENERAL_BURST = 60


def test_a_general_ceiling_applies_per_visitor_across_the_whole_api():
    visitor = {"X-Forwarded-For": "203.0.113.20"}
    # Spend the allowance over a mix of endpoints: the ceiling is one bucket for
    # the whole API, not a fresh allowance for each route.
    spent = []
    for i in range(GENERAL_BURST):
        if i % 2:
            spent.append(client.get("/api/meta", headers=visitor).status_code)
        else:
            spent.append(
                client.get(
                    "/api/top-names", params={"sex": "F", "year": 2015}, headers=visitor
                ).status_code
            )
    assert 429 not in spent

    assert client.get("/api/meta", headers=visitor).status_code == 429
    assert client.get("/api/names/emma", params={"sex": "F"}, headers=visitor).status_code == 429

    # A visitor at another address still has their own full allowance.
    assert client.get("/api/meta", headers={"X-Forwarded-For": "198.51.100.8"}).status_code == 200


def test_the_backend_refuses_to_serve_in_production_with_no_secret_configured(monkeypatch):
    from app import config

    monkeypatch.setattr(config, "BACKEND_SHARED_SECRET", None)
    monkeypatch.setattr(config, "IS_PRODUCTION", True)
    # A deployment that lost its secret must fail closed rather than serve the
    # public internet without a guard.
    assert anonymous.get("/api/meta").status_code == 401
    assert client.get("/api/meta").status_code == 401
    assert anonymous.get("/api/health").status_code == 200


def test_local_development_without_a_secret_stays_usable(monkeypatch):
    from app import config

    # A fresh checkout has no secret configured and `make dev` must still work.
    monkeypatch.setattr(config, "BACKEND_SHARED_SECRET", None)
    monkeypatch.setattr(config, "IS_PRODUCTION", False)
    assert anonymous.get("/api/meta").status_code == 200


def test_an_unauthenticated_request_never_spends_a_visitors_allowance(monkeypatch):
    _no_provider_calls(monkeypatch)
    spoofed = {"X-Forwarded-For": "203.0.113.50"}

    for _ in range(GENERAL_BURST + CHAT_BURST):
        assert (
            anonymous.post("/api/chat", json={"message": "hi"}, headers=spoofed).status_code == 401
        )

    # The secret is checked before the forwarded address is believed, so an
    # outsider cannot lock a visitor out by claiming to be them.
    assert client.post("/api/chat", json={"message": "hi"}, headers=spoofed).status_code == 503


def test_a_request_with_no_forwarded_address_is_bucketed_by_its_peer_address(monkeypatch):
    _no_provider_calls(monkeypatch)

    for _ in range(CHAT_BURST):
        client.post("/api/chat", json={"message": "hi"})
    assert client.post("/api/chat", json={"message": "hi"}).status_code == 429

    # That bucket belongs to the caller itself, not to any forwarded visitor.
    assert (
        client.post(
            "/api/chat", json={"message": "hi"}, headers={"X-Forwarded-For": "203.0.113.30"}
        ).status_code
        == 503
    )


def test_the_health_probe_is_never_rate_limited():
    probe = {"X-Forwarded-For": "203.0.113.40"}
    for _ in range(GENERAL_BURST + 5):
        assert anonymous.get("/api/health", headers=probe).status_code == 200


def test_chat_rejects_an_oversized_history():
    # `message` is capped at 500 characters, but `history` was not capped at
    # all - and it is client-controlled, so it is what an abusive caller grows
    # to spend the provider quota. Only the last few entries are ever read.
    response = client.post(
        "/api/chat",
        json={
            "message": "How many names?",
            "history": [{"role": "user", "content": "x" * 50_000}],
        },
    )
    assert response.status_code == 422


def test_chat_rejects_too_many_history_entries():
    response = client.post(
        "/api/chat",
        json={
            "message": "How many names?",
            "history": [{"role": "user", "content": "hi"} for _ in range(500)],
        },
    )
    assert response.status_code == 422


def test_forecast_model_section_is_the_pooled_model_card():
    """The served forecast has to say what actually produced it.

    One pooled model forecasts every name, so what the page can honestly
    report about "the model" is a property of the batch, not of this name:
    the model class, what it was trained on, and the features it reads. An
    ARIMA order or a residual-diagnostics panel would be describing a per-name
    fit that no longer happens. See docs/adr/0010-a-pooled-model-replaces-per-name-arima.md.
    """
    body = client.get("/api/names/emma/forecast", params={"sex": "F"}).json()
    model = body["model"]

    assert "LightGBM" in model["model_name"]
    assert model["horizons"] == 5
    assert model["features"]
    assert model["trained_through"] == body["history"][-1]["year"]
    assert "order" not in model
    assert "diagnostics" not in model


def test_forecast_survives_an_artifact_published_before_the_model_card(sample_db, monkeypatch):
    """A missing model card costs the panel, not the endpoint.

    The database is published independently of this code (ADR 0006), so a
    deploy can meet an artifact whose batch predates `model_card` and has no
    such table at all. The forecast itself is still stored and still servable,
    and the page already renders without the panel.
    """
    import shutil

    from app import config, database

    stale = Path(sample_db).parent / "stale.db"
    shutil.copyfile(sample_db, stale)
    conn = sqlite3.connect(stale)
    conn.execute("DROP TABLE model_card")
    conn.commit()
    conn.close()

    monkeypatch.setattr(config, "NAMES_DB_PATH", str(stale))
    database.resolve_database_path.cache_clear()
    try:
        response = client.get("/api/names/emma/forecast", params={"sex": "F"})
    finally:
        database.resolve_database_path.cache_clear()

    assert response.status_code == 200
    body = response.json()
    assert len(body["forecast"]) == 5
    assert body["model"] is None


def test_a_forecast_from_an_artifact_that_predates_the_strata_still_serves(
    sample_db, tmp_path, monkeypatch
):
    """The database is published independently of this code (ADR 0006).

    So a deploy can meet an artifact built before `forecasts` carried a
    stratum and before `calibration` was keyed by one. That should cost the
    band label, exactly as an artifact without a `model_card` costs the model
    panel — not the forecast, and not the endpoint.
    """
    from app import config, database

    legacy = tmp_path / "legacy.db"
    legacy.write_bytes(Path(sample_db).read_bytes())
    conn = sqlite3.connect(str(legacy))
    conn.execute("ALTER TABLE forecasts DROP COLUMN tier")
    conn.execute("ALTER TABLE forecasts DROP COLUMN volatility_bin")
    conn.execute("DROP TABLE calibration")
    conn.execute(
        "CREATE TABLE calibration (nominal_level REAL NOT NULL PRIMARY KEY, "
        "empirical_coverage REAL NOT NULL, n INTEGER NOT NULL)"
    )
    conn.execute("INSERT INTO calibration VALUES (0.8, 0.44, 45)")
    conn.commit()
    conn.close()

    monkeypatch.setattr(config, "NAMES_DB_PATH", str(legacy))
    database.resolve_database_path.cache_clear()
    try:
        response = client.get("/api/names/emma/forecast", params={"sex": "F"})
    finally:
        database.resolve_database_path.cache_clear()

    assert response.status_code == 200
    body = response.json()
    assert body["forecast"]
    assert body["calibration"] == {}
    assert body["stratum"] is None
