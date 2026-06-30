from pathlib import Path
from typing import Dict, List, Tuple, Any
import json

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "feeds-1.csv"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"

PM25_MODEL_PATH = MODELS_DIR / "pm25_model.pkl"
TEMP_MODEL_PATH = MODELS_DIR / "temperature_model.pkl"
HUMIDITY_MODEL_PATH = MODELS_DIR / "humidity_model.pkl"

FORECAST_DAYS = 7
FEATURE_COLS = ["pm25", "temperature", "humidity"]


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {file_path}")

    df = pd.read_csv(file_path)

    required = ["created_at", "field1", "field2", "field3"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom dataset tidak lengkap: {missing}")

    df = df.copy()
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df["field1"] = pd.to_numeric(df["field1"], errors="coerce")
    df["field2"] = pd.to_numeric(df["field2"], errors="coerce")
    df["field3"] = pd.to_numeric(df["field3"], errors="coerce")

    df = df.rename(
        columns={
            "field1": "pm25",
            "field2": "temperature",
            "field3": "humidity",
        }
    )

    df = df.dropna(subset=["created_at", "pm25", "temperature", "humidity"])
    df = df.sort_values("created_at").reset_index(drop=True)
    return df


def extract_model(obj: Any) -> Any:
    if hasattr(obj, "predict"):
        return obj

    if isinstance(obj, dict):
        for key in ["model", "xgb_model", "estimator", "pipeline"]:
            if key in obj and hasattr(obj[key], "predict"):
                return obj[key]

        for value in obj.values():
            if hasattr(value, "predict"):
                return value

    raise TypeError(f"Isi file model tidak valid: {type(obj)}")


def load_models() -> Tuple[Any, Any, Any]:
    for path in [PM25_MODEL_PATH, TEMP_MODEL_PATH, HUMIDITY_MODEL_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Model tidak ditemukan: {path}")

    pm25_model = extract_model(joblib.load(PM25_MODEL_PATH))
    temperature_model = extract_model(joblib.load(TEMP_MODEL_PATH))
    humidity_model = extract_model(joblib.load(HUMIDITY_MODEL_PATH))

    return pm25_model, temperature_model, humidity_model


def get_last_known_state(df: pd.DataFrame) -> Dict[str, float]:
    last_row = df[FEATURE_COLS].iloc[-1]
    return {
        "pm25": float(last_row["pm25"]),
        "temperature": float(last_row["temperature"]),
        "humidity": float(last_row["humidity"]),
    }


def build_feature_row(state: Dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame([state], columns=FEATURE_COLS)


def predict_one_step(models: Tuple[Any, Any, Any], feature_row: pd.DataFrame) -> Dict[str, float]:
    pm25_model, temperature_model, humidity_model = models

    pm25_pred = float(pm25_model.predict(feature_row)[0])
    temp_pred = float(temperature_model.predict(feature_row)[0])
    humidity_pred = float(humidity_model.predict(feature_row)[0])

    return {
        "pm25": pm25_pred,
        "temperature": temp_pred,
        "humidity": humidity_pred,
    }


def recursive_forecast(
    initial_state: Dict[str, float],
    models: Tuple[Any, Any, Any],
    days: int = 7
) -> List[Dict[str, float]]:
    current_state = initial_state.copy()
    results = []

    for step in range(1, days + 1):
        feature_row = build_feature_row(current_state)
        prediction = predict_one_step(models, feature_row)

        results.append(
            {
                "day_ahead": f"D+{step}",
                "pm25_prediction": round(prediction["pm25"], 3),
                "temperature_prediction": round(prediction["temperature"], 3),
                "humidity_prediction": round(prediction["humidity"], 3),
            }
        )

        current_state = prediction

    return results


def save_results(results: List[Dict[str, float]]) -> None:
    ensure_output_dir()

    csv_path = OUTPUT_DIR / "forecast_results.csv"
    json_path = OUTPUT_DIR / "forecast_results.json"

    pd.DataFrame(results).to_csv(csv_path, index=False)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


def predict_future(days: int = 7) -> List[Dict[str, float]]:
    df = load_dataset(DATASET_PATH)
    models = load_models()
    initial_state = get_last_known_state(df)
    results = recursive_forecast(initial_state, models, days=days)
    save_results(results)
    return results


def main() -> None:
    results = predict_future(days=FORECAST_DAYS)
    for item in results:
        print(item)


if __name__ == "__main__":
    main()