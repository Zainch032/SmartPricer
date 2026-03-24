from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import pickle
import numpy as np
import pandas as pd

app = FastAPI(title="Phone Price Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------
# Load model and ColumnTransformer
# ---------------------------------------------------------------
try:
    model = pickle.load(open("model/model.pkl", "rb"))
    ct    = pickle.load(open("model/ct.pkl", "rb"))
    print("✅ Model and CT loaded successfully")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load model/ct: {e}")


# ---------------------------------------------------------------
# Input schema — exactly matches X_raw columns
# ---------------------------------------------------------------
class PhoneFeatures(BaseModel):
    Brand:           str
    Has_5G:          int
    Has_NFC:         int
    Has_IR_Blaster:  int
    Processor_Model: str
    Processor_Core:  float
    Processor_Speed: float
    rating:          float
    Ram:             float
    Memory:          float
    Battery:         float
    Screen_size:     float
    Resolution:      str
    Num_Back_Cam:    float
    Back_Camera_MP:  float
    Front_CAM_MP:    float
    Num_Front_Cam:   float
    Refresh_Rate:    float
    Fast_Charge:     float
    OS_Name:         str
    os_version:      float
    card_support:    int
    slot_type:       str
    capacity_gb:     float


# ---------------------------------------------------------------
# Preprocess — column order must match X_raw exactly
# ---------------------------------------------------------------
def preprocess(data: PhoneFeatures) -> np.ndarray:
    row = {
        "Brand":           data.Brand,
        "Has_5G":          data.Has_5G,
        "Has_NFC":         data.Has_NFC,
        "Has_IR_Blaster":  data.Has_IR_Blaster,
        "Processor_Model": data.Processor_Model,
        "Processor_Core":  data.Processor_Core,
        "Processor_Speed": data.Processor_Speed,
        "rating":          data.rating,
        "Ram":             data.Ram,
        "Memory":          data.Memory,
        "Battery":         data.Battery,
        "Screen_size":     data.Screen_size,
        "Resolution":      data.Resolution,
        "Num_Back_Cam":    data.Num_Back_Cam,
        "Back_Camera_MP":  data.Back_Camera_MP,
        "Front_CAM_MP":    data.Front_CAM_MP,
        "Num_Front_Cam":   data.Num_Front_Cam,
        "Refresh_Rate":    data.Refresh_Rate,
        "Fast_Charge":     data.Fast_Charge,
        "OS_Name":         data.OS_Name,
        "os_version":      data.os_version,
        "card_support":    data.card_support,
        "slot_type":       data.slot_type,
        "capacity_gb":     data.capacity_gb,
    }

    raw_row    = pd.DataFrame([row])
    scaled_row = ct.transform(raw_row)
    return scaled_row


# ---------------------------------------------------------------
# Routes
# ---------------------------------------------------------------
@app.get("/")
def root():
    return FileResponse("index.html")


@app.post("/predict")
def predict(features: PhoneFeatures):
    try:
        scaled_row      = preprocess(features)
        prediction      = model.predict(scaled_row)
        predicted_price = float(prediction[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return {
        "predicted_price": round(predicted_price, 2),
        "currency": "INR",
    }


@app.get("/processor-data")
def get_processor_data():
    try:
        df     = pd.read_csv("data.csv")
        groups = (
            df.groupby("Processor_Name")[["Processor_Model", "Processor_Core", "Processor_Speed"]]
            .first()
            .reset_index()
        )
        result = {}
        for _, row in groups.iterrows():
            brand = str(row["Processor_Model"])
            if brand not in result:
                result[brand] = []
            result[brand].append({
                "name":  str(row["Processor_Name"]),
                "model": brand,
                "core":  float(row["Processor_Core"]),
                "speed": float(row["Processor_Speed"]),
            })
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load processor data: {str(e)}")
