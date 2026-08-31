from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["database"] == "ok"


def test_meta():
    response = client.get("/api/meta")
    assert response.status_code == 200
    assert response.json() == {"min_year": 1960, "max_year": 2024}


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
