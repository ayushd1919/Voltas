from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    # ok when artifacts loaded, degraded when not (still acceptable)
    assert r.json()["status"] in {"ok", "degraded"}


def test_predict_shape_when_artifacts_present():
    # This will pass only if artifacts are present (CI pulls via DVC or committed)
    r = client.get("/health")
    if r.json()["status"] != "ok":
        return  # skip gracefully if artifacts weren’t pulled
    body = {
        "records": [
            {
                "product_id": "AC-001",
                "price_inr": 40990,
                "city": "Mumbai",
                "platform": "Croma",
                "energy_rating_stars": 5,
                "warranty_years": 2,
                "capacity_unified": 1.5,
                "price_density_score": 0.62,
                "warranty_quality_interaction": 1.24,
            }
        ]
    }
    r = client.post("/predict", json=body)
    assert r.status_code == 200
    data = r.json()
    assert "predictions" in data and "labels" in data
