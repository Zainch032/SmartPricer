"""Phone price API. 4 required answers, 6 optional ones. Run: uvicorn app.main:app --reload"""
import os
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
bundle = joblib.load(ROOT / "model" / "model.joblib")
model, REQUIRED, OPTIONAL = bundle["model"], bundle["required"], bundle["optional"]
BANDS, BRANDS = bundle["bands"], bundle["brands"]
INR_TO_PKR = float(os.getenv("INR_TO_PKR", "3.3"))  # set to today's rate; the model itself predicts INR

app = FastAPI(title="Phone Price Predictor")


class PhoneIn(BaseModel):
    # --- required: all visible in Settings > About phone ---
    Brand: str
    Ram: float = Field(gt=0, le=32)
    Memory: float = Field(gt=0, le=2048)
    Has_5G: bool
    # --- optional: leave out if you don't know ---
    Screen_size: Optional[float] = Field(None, gt=2, lt=12)
    Battery: Optional[float] = Field(None, gt=500, lt=15000)
    Back_Camera_MP: Optional[float] = Field(None, gt=0, le=250)
    Has_NFC: Optional[bool] = None
    Fast_Charge: Optional[float] = Field(None, ge=0, le=300)   # 0 = no fast charging
    os_version: Optional[float] = Field(None, ge=1, le=30)


@app.get("/")
def index():
    return FileResponse(ROOT / "app" / "index.html")


@app.get("/options")
def options():
    return {"brands": BRANDS, "inr_to_pkr": INR_TO_PKR}


@app.post("/predict")
def predict(p: PhoneIn):
    row = p.model_dump()
    row["Has_5G"] = int(row["Has_5G"])
    if row["Has_NFC"] is not None:
        row["Has_NFC"] = int(row["Has_NFC"])
    if row["Brand"] not in BRANDS:
        row["Brand"] = "Other"
    X = pd.DataFrame([row])[REQUIRED + OPTIONAL].astype(object).where(lambda d: d.notna(), np.nan)
    for c in X.columns:
        if c != "Brand":
            X[c] = pd.to_numeric(X[c])
    try:
        price = float(model.predict(X)[0])
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {e}")

    answered = sum(row[c] is not None for c in OPTIONAL)
    band = BANDS["detailed" if answered >= 3 else "basic"]
    lo, hi = price * np.exp(-band["hi"]), price * np.exp(-band["lo"])   # band is log(pred/actual)
    return {
        "price_inr": round(price, -2), "low_inr": round(lo, -2), "high_inr": round(hi, -2),
        "price_pkr": round(price * INR_TO_PKR, -2), "low_pkr": round(lo * INR_TO_PKR, -2),
        "high_pkr": round(hi * INR_TO_PKR, -2),
        "optional_answered": answered, "optional_total": len(OPTIONAL),
    }
