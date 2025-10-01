# Voltas Availability Prediction API

FastAPI service that serves a trained scikit-learn pipeline to predict **availability** (0=In Stock, 1=Out of Stock). Ships with Docker.

## Endpoints
- `GET /health` — model & artifact status
- `POST /predict` — JSON `{ "records": [ { ...features } ] }` → predictions, labels, probabilities
- `GET /docs` — interactive Swagger UI

## Project Structure
voltas_api/
├─ app/
│ ├─ main.py
│ ├─ artifacts/
│ │ ├─ inference_pipeline.joblib
│ │ └─ expected_columns.json
├─ requirements.txt
├─ Dockerfile
├─ test_request.json
└─ README.md

## Prereqs
- Python 3.11 (recommended) or Docker Desktop
- Artifacts: `app/artifacts/inference_pipeline.joblib` and `expected_columns.json`

## Run (no Docker)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Health: http://127.0.0.1:8000/health

Docs: http://127.0.0.1:8000/docs

## Test (PowerShell)
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health"
Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method POST -ContentType "application/json" -Body (Get-Content .\test_request.json -Raw)
```

## Run with Docker
```powershell
docker build -t voltas-api:latest .
docker run --rm -p 8000:8000 voltas-api:latest
```
Health: http://127.0.0.1:8000/health

Docs: http://127.0.0.1:8000/docs

## Example Request (test_request.json)
```json
{
  "records": [
    {
      "product_id": "AC-001",
      "product_category": "Air Conditioner",
      "sub_type": "Split",
      "model_name": "CoolX 1.5T",
      "capacity_tons": 1.5,
      "energy_rating_stars": 5,
      "color": "White",
      "price_inr": 40990,
      "manufacturing_date": "2024-03-01",
      "warranty_years": 2,
      "customer_rating": 4.4,
      "city": "Mumbai",
      "platform": "Croma",
      "discount_offered": 12,
      "warranty_duration_months": 24,
      "review_sentiment": "positive",
      "return_status": "No Return",
      "resolved_status": "Resolved",
      "review_date": "2024-05-10",
      "reviewer_location": "Mumbai",
      "product_name": "CoolX 1.5T Inverter",
      "username": "user123",
      "capacity_unified": 1.5,
      "price_density_score": 0.62,
      "warranty_quality_interaction": 1.24
    }
  ]
}