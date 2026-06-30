import os
import sys
import sqlite3
import threading
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# ======================================================
# Konfigurasi Logging
# ======================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ======================================================
# Penentuan Path Dasar (Railway Compatible)
# ======================================================
BASE_DIR = Path(__file__).resolve().parent.parent  # DEPLOY/
sys.path.insert(0, str(BASE_DIR))

# ======================================================
# Import Fungsi Prediksi (dengan fallback jika belum ada)
# ======================================================
try:
    from predict_model import predict_future
    PREDICT_MODEL_AVAILABLE = True
    logger.info("Modul predict_model berhasil diimpor.")
except ImportError:
    PREDICT_MODEL_AVAILABLE = False
    logger.warning("Modul predict_model tidak ditemukan. Endpoint /api/predict akan dinonaktifkan.")
    predict_future = None  # type: ignore

# ======================================================
# Validasi Keberadaan Folder dan File Penting
# ======================================================
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
INDEX_HTML = TEMPLATES_DIR / "index.html"

if not TEMPLATES_DIR.exists():
    raise RuntimeError(
        f"Folder templates tidak ditemukan di {TEMPLATES_DIR}. "
        "Pastikan struktur project sudah benar."
    )
if not INDEX_HTML.exists():
    raise RuntimeError(
        f"File index.html tidak ditemukan di {INDEX_HTML}. "
        "Pastikan file dashboard sudah tersedia."
    )
if not STATIC_DIR.exists():
    logger.warning(f"Folder static tidak ditemukan di {STATIC_DIR}. File CSS/JS tidak akan termuat.")

# ======================================================
# Inisialisasi Aplikasi FastAPI
# ======================================================
app = FastAPI(
    title="Air Quality Monitoring & Prediction",
    version="1.0.0",
    description="Sistem monitoring kualitas udara real-time dan prediksi menggunakan XGBoost",
)

# ======================================================
# Mount Static Files dan Templates
# ======================================================
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# ======================================================
# Konfigurasi Database
# ======================================================
DATABASE_DIR = BASE_DIR / "database"
DATABASE_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATABASE_DIR / "history.db"

# Global connection & lock untuk thread-safe
_db_connection: Optional[sqlite3.Connection] = None
_db_lock = threading.Lock()

def get_db() -> sqlite3.Connection:
    """
    Mendapatkan koneksi database yang sudah diinisialisasi (thread-safe).
    Membuat tabel jika belum ada.
    """
    global _db_connection
    if _db_connection is None:
        with _db_lock:
            if _db_connection is None:
                _db_connection = sqlite3.connect(str(DB_PATH), check_same_thread=False)
                _db_connection.row_factory = sqlite3.Row
                # Aktifkan WAL mode untuk performa lebih baik
                _db_connection.execute("PRAGMA journal_mode=WAL;")
                _create_tables(_db_connection)
                logger.info("Database berhasil diinisialisasi dan tabel siap.")
    return _db_connection

def _create_tables(conn: sqlite3.Connection) -> None:
    """
    Membuat tabel-tabel yang diperlukan jika belum ada.
    """
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sensor_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            pm25 REAL NOT NULL,
            temperature REAL NOT NULL,
            humidity REAL NOT NULL
        );
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            target_date DATE NOT NULL,
            pm25_prediction REAL,
            temperature_prediction REAL,
            humidity_prediction REAL,
            pm25_actual REAL,
            temperature_actual REAL,
            humidity_actual REAL,
            pm25_error REAL,
            temperature_error REAL,
            humidity_error REAL
        );
    """)
    conn.commit()
    logger.info("Struktur tabel diverifikasi.")

# ======================================================
# Model Pydantic untuk Request Body
# ======================================================
class SensorData(BaseModel):
    pm25: float
    temperature: float
    humidity: float

class PredictRequest(BaseModel):
    days: int

# ======================================================
# Event Startup
# ======================================================
@app.on_event("startup")
async def startup_event():
    """Inisialisasi database connection dan laporan status."""
    get_db()
    logger.info("Aplikasi siap menerima request.")

# ======================================================
# Endpoint: Halaman Utama (Dashboard)
# ======================================================
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    Menampilkan halaman dashboard monitoring dan prediksi.
    """
    try:
        template = templates.get_template("index.html")
        rendered = template.render({"request": request})
        return HTMLResponse(content=rendered)
    except Exception as e:
        logger.error(f"Gagal merender template index.html: {e}")
        raise HTTPException(status_code=500, detail="Template rendering error")

# ======================================================
# Endpoint: Menerima Data Sensor dari ESP32
# ======================================================
@app.post("/api/sensor")
async def receive_sensor_data(data: SensorData):
    """
    Menerima data sensor dari ESP32 dan menyimpannya ke database.
    """
    conn = get_db()
    try:
        with _db_lock:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO sensor_data (pm25, temperature, humidity) VALUES (?, ?, ?);",
                (data.pm25, data.temperature, data.humidity),
            )
            conn.commit()
            logger.info(
                f"Data sensor disimpan: PM2.5={data.pm25}, Suhu={data.temperature}, Kelembapan={data.humidity}"
            )
        return {
            "status": "success",
            "message": "Sensor data berhasil disimpan.",
        }
    except Exception as e:
        logger.error(f"Gagal menyimpan data sensor: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# ======================================================
# Endpoint: Data Sensor Terbaru (Konsisten dengan format wrapper)
# ======================================================
@app.get("/api/latest")
async def get_latest_sensor():
    """
    Mengambil data sensor terbaru.
    Selalu mengembalikan format {status, data} agar kompatibel dengan frontend.
    """
    conn = get_db()
    try:
        with _db_lock:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sensor_data ORDER BY id DESC LIMIT 1;")
            row = cursor.fetchone()
        if row is None:
            # Tidak ada data: kembalikan data kosong dengan status success
            return {
                "status": "success",
                "data": {
                    "created_at": None,
                    "pm25": 0.0,
                    "temperature": 0.0,
                    "humidity": 0.0,
                }
            }
        return {
            "status": "success",
            "data": {
                "created_at": row["created_at"],
                "pm25": row["pm25"],
                "temperature": row["temperature"],
                "humidity": row["humidity"],
            }
        }
    except Exception as e:
        logger.error(f"Gagal mengambil data sensor terbaru: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# ======================================================
# Endpoint: Riwayat Data Sensor
# ======================================================
@app.get("/api/history")
async def get_sensor_history(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """
    Mengambil riwayat data sensor dengan paginasi.
    """
    conn = get_db()
    try:
        with _db_lock:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM sensor_data ORDER BY id DESC LIMIT ? OFFSET ?;",
                (limit, offset),
            )
            rows = cursor.fetchall()
        history = [
            {
                "id": row["id"],
                "created_at": row["created_at"],
                "pm25": row["pm25"],
                "temperature": row["temperature"],
                "humidity": row["humidity"],
            }
            for row in rows
        ]
        return {
            "status": "success",
            "data": history,
        }
    except Exception as e:
        logger.error(f"Gagal mengambil riwayat sensor: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# ======================================================
# Endpoint: Prediksi Kualitas Udara
# ======================================================
@app.post("/api/predict")
async def predict_air_quality(req: PredictRequest):
    """
    Melakukan prediksi kualitas udara untuk D+1 hingga D+7.
    """
    if not PREDICT_MODEL_AVAILABLE or predict_future is None:
        raise HTTPException(
            status_code=503,
            detail="Modul prediksi tidak tersedia. Pastikan predict_model.py sudah benar.",
        )

    days = req.days
    if days < 1 or days > 7:
        raise HTTPException(status_code=400, detail="Jumlah hari harus antara 1 sampai 7.")

    try:
        predictions: List[Dict[str, Any]] = predict_future(days)
        if not isinstance(predictions, list) or len(predictions) == 0:
            raise ValueError("Fungsi predict_future harus mengembalikan list non-kosong.")

        conn = get_db()
        saved_predictions = []
        prediction_date = datetime.now().isoformat()

        with _db_lock:
            cursor = conn.cursor()
            for pred in predictions:
                target_date = pred.get("target_date")
                pm25_pred = pred.get("pm25")
                temp_pred = pred.get("temperature")
                hum_pred = pred.get("humidity")

                cursor.execute(
                    """INSERT INTO prediction_history 
                    (prediction_date, target_date, pm25_prediction, temperature_prediction, humidity_prediction)
                    VALUES (?, ?, ?, ?, ?);""",
                    (prediction_date, target_date, pm25_pred, temp_pred, hum_pred),
                )
                saved_predictions.append({
                    "prediction_date": prediction_date,
                    "target_date": target_date,
                    "pm25_prediction": pm25_pred,
                    "temperature_prediction": temp_pred,
                    "humidity_prediction": hum_pred,
                })
            conn.commit()

        logger.info(f"Prediksi {days} hari berhasil disimpan.")
        return {
            "status": "success",
            "message": f"Prediksi untuk {days} hari ke depan berhasil dihitung.",
            "data": saved_predictions,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Gagal melakukan prediksi: {e}")
        raise HTTPException(status_code=500, detail=f"Prediksi gagal: {str(e)}")

# ======================================================
# Endpoint: Riwayat Prediksi
# ======================================================
@app.get("/api/predictions")
async def get_prediction_history(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """
    Mengambil seluruh riwayat prediksi yang telah dilakukan.
    """
    conn = get_db()
    try:
        with _db_lock:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM prediction_history ORDER BY id DESC LIMIT ? OFFSET ?;",
                (limit, offset),
            )
            rows = cursor.fetchall()
        predictions = [
            {
                "id": row["id"],
                "prediction_date": row["prediction_date"],
                "target_date": row["target_date"],
                "pm25_prediction": row["pm25_prediction"],
                "temperature_prediction": row["temperature_prediction"],
                "humidity_prediction": row["humidity_prediction"],
                "pm25_actual": row["pm25_actual"],
                "temperature_actual": row["temperature_actual"],
                "humidity_actual": row["humidity_actual"],
                "pm25_error": row["pm25_error"],
                "temperature_error": row["temperature_error"],
                "humidity_error": row["humidity_error"],
            }
            for row in rows
        ]
        return {
            "status": "success",
            "data": predictions,
        }
    except Exception as e:
        logger.error(f"Gagal mengambil riwayat prediksi: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# ======================================================
# Entry Point untuk Lokal / Railway
# ======================================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)