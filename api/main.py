from pathlib import Path
from typing import List, Optional
from datetime import datetime, timedelta
import pandas as pd
import sqlite3
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

from predict_model import predict_future, load_dataset, DATASET_PATH


BASE_DIR = Path(__file__).resolve().parent.parent
DB_DIR = BASE_DIR / "database"
DB_PATH = DB_DIR / "history.db"

app = FastAPI(
    title="Air Quality Prediction API",
    version="1.0.0",
    description="API monitoring, prediksi, dan riwayat kualitas udara berbasis XGBoost"
)


class PredictRequest(BaseModel):
    days: int = Field(..., ge=1, le=7)


class LatestResponse(BaseModel):
    created_at: Optional[str]
    pm25: Optional[float]
    temperature: Optional[float]
    humidity: Optional[float]


class PredictionItem(BaseModel):
    day_ahead: str
    pm25_prediction: float
    temperature_prediction: float
    humidity_prediction: float


class HistoryItem(BaseModel):
    id: int
    prediction_date: str
    target_date: str
    pm25_prediction: float
    temperature_prediction: float
    humidity_prediction: float
    pm25_actual: Optional[float]
    temperature_actual: Optional[float]
    humidity_actual: Optional[float]
    pm25_error: Optional[float]
    temperature_error: Optional[float]
    humidity_error: Optional[float]


def get_connection():
    DB_DIR.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_date TEXT NOT NULL,
            target_date TEXT NOT NULL,
            pm25_prediction REAL NOT NULL,
            temperature_prediction REAL NOT NULL,
            humidity_prediction REAL NOT NULL,
            pm25_actual REAL,
            temperature_actual REAL,
            humidity_actual REAL,
            pm25_error REAL,
            temperature_error REAL,
            humidity_error REAL
        )
        """
    )
    conn.commit()
    conn.close()


def save_history(predictions: List[dict]):
    conn = get_connection()
    cursor = conn.cursor()

    for item in predictions:
        cursor.execute(
            """
            INSERT INTO prediction_history (
                prediction_date, target_date,
                pm25_prediction, temperature_prediction, humidity_prediction,
                pm25_actual, temperature_actual, humidity_actual,
                pm25_error, temperature_error, humidity_error
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item["prediction_date"],
                item["target_date"],
                item["pm25_prediction"],
                item["temperature_prediction"],
                item["humidity_prediction"],
                None,
                None,
                None,
                None,
                None,
                None,
            )
        )

    conn.commit()
    conn.close()


def compute_target_date(base_date: datetime, day_ahead: str) -> str:
    day_number = int(day_ahead.replace("D+", ""))
    target = base_date + timedelta(days=day_number)
    return target.strftime("%Y-%m-%d")


@app.on_event("startup")
def startup_event():
    init_db()


@app.get("/latest", response_model=LatestResponse)
def get_latest():
    try:
        df = load_dataset(DATASET_PATH)
        last = df.tail(1).iloc[0]
        return {
            "created_at": str(last["created_at"]),
            "pm25": float(last["pm25"]),
            "temperature": float(last["temperature"]),
            "humidity": float(last["humidity"]),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=List[PredictionItem])
def predict(payload: PredictRequest):
    try:
        results = predict_future(days=payload.days)

        df = load_dataset(DATASET_PATH)
        prediction_date = df.tail(1).iloc[0]["created_at"]
        if pd.isna(prediction_date):
            raise ValueError("Tanggal prediksi tidak valid")

        if hasattr(prediction_date, "to_pydatetime"):
            prediction_date = prediction_date.to_pydatetime()

        prediction_date_str = prediction_date.strftime("%Y-%m-%d %H:%M:%S")

        history_rows = []
        for item in results:
            history_rows.append(
                {
                    "prediction_date": prediction_date_str,
                    "target_date": compute_target_date(prediction_date, item["day_ahead"]),
                    "pm25_prediction": item["pm25_prediction"],
                    "temperature_prediction": item["temperature_prediction"],
                    "humidity_prediction": item["humidity_prediction"],
                }
            )

        save_history(history_rows)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history", response_model=List[HistoryItem])
def get_history():
    try:
        conn = get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT
                id, prediction_date, target_date,
                pm25_prediction, temperature_prediction, humidity_prediction,
                pm25_actual, temperature_actual, humidity_actual,
                pm25_error, temperature_error, humidity_error
            FROM prediction_history
            ORDER BY id DESC
            """
        )
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("api.main:app", host="0.0.0.0", port=port, reload=True)