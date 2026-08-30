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
