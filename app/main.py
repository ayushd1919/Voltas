from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import joblib
import json
import os

APP_DIR = os.path.dirname(__file__)
ART_DIR = os.path.join(APP_DIR, "artifacts")

PIPE_PATH = os.path.join(ART_DIR, "inference_pipeline.joblib")
COLS_PATH = os.path.join(ART_DIR, "expected_columns.json")

# load pipeline + expected columns on startup
inference_pipeline = joblib.load(PIPE_PATH)
expected = json.load(open(COLS_PATH))
EXPECTED_COLS = expected["expected_input_cols"]

app = FastAPI(title="Voltas Availability API", version="1.0")


class PredictRequest(BaseModel):
    records: List[Dict[str, Any]]


class PredictResponse(BaseModel):
    predictions: List[int]
    labels: List[str]
    probabilities: Optional[List[float]] = None


LABEL_MAP = {0: "In Stock", 1: "Out of Stock"}


@app.get("/health")
def health():
    return {"status": "ok", "expected_features": len(EXPECTED_COLS)}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.records:
        raise HTTPException(400, "No records provided.")
    df = pd.DataFrame(req.records)

    # align columns: keep only expected, add missing as NaN
    for c in EXPECTED_COLS:
        if c not in df.columns:
            df[c] = None
    df = df[EXPECTED_COLS]  # reorder & drop extras

    # run inference
    try:
        proba = None
        if hasattr(inference_pipeline.named_steps["model"], "predict_proba"):
            proba = inference_pipeline.predict_proba(df)[:, 1].tolist()
        preds = inference_pipeline.predict(df).tolist()
    except Exception as e:
        raise HTTPException(500, f"Inference error: {e}")

    labels = [LABEL_MAP.get(int(p), str(p)) for p in preds]
    return PredictResponse(
        predictions=[int(p) for p in preds], labels=labels, probabilities=proba
    )
