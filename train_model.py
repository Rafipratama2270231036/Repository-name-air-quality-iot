from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
# prefer dataset/feeds-1.csv; fall back to feeds-1.csv at project root
_candidate = BASE_DIR / "dataset" / "feeds-1.csv"
if _candidate.exists():
    DATASET_PATH = _candidate
else:
    DATASET_PATH = BASE_DIR / "feeds-1.csv"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"
LAG = 3
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_STATE = 42


@dataclass
class ModelResult:
    target: str
    metrics: dict
    model_path: str


def ensure_directories() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    required_cols = ["created_at", "field1", "field2", "field3"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {missing}")

    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    df["field1"] = pd.to_numeric(df["field1"], errors="coerce")
    df["field2"] = pd.to_numeric(df["field2"], errors="coerce")
    df["field3"] = pd.to_numeric(df["field3"], errors="coerce")

    df = df.dropna(subset=["created_at", "field1", "field2", "field3"]).copy()
    df = df.sort_values("created_at").drop_duplicates(subset=["created_at"], keep="last")
    df = df.rename(
        columns={
            "field1": "pm25",
            "field2": "temperature",
            "field3": "humidity",
        }
    ).reset_index(drop=True)

    return df


def create_lag_features(df: pd.DataFrame, target_col: str, lag: int = LAG) -> pd.DataFrame:
    data = df[["created_at", "pm25", "temperature", "humidity"]].copy()

    for i in range(1, lag + 1):
        data[f"{target_col}_lag_{i}"] = data[target_col].shift(i)

    data["target"] = data[target_col]
    data = data.dropna().reset_index(drop=True)

    feature_cols = [f"{target_col}_lag_{i}" for i in range(1, lag + 1)]
    X = data[feature_cols]
    y = data["target"]

    result = data[["created_at"]].copy()
    for col in feature_cols:
        result[col] = X[col]
    result["target"] = y

    return result


def time_series_split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(data)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    train = data.iloc[:train_end].copy()
    val = data.iloc[train_end:val_end].copy()
    test = data.iloc[val_end:].copy()

    return train, val, test


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    y_true_safe = np.where(y_true == 0, 1e-8, y_true)
    mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100

    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
        "R2": float(r2),
    }


def train_one_model(target_name: str, df: pd.DataFrame) -> ModelResult:
    lagged = create_lag_features(df, target_name, LAG)
    train_df, val_df, test_df = time_series_split(lagged)

    feature_cols = [f"{target_name}_lag_{i}" for i in range(1, LAG + 1)]

    X_train = train_df[feature_cols].values
    y_train = train_df["target"].values

    X_val = val_df[feature_cols].values
    y_val = val_df["target"].values

    X_test = test_df[feature_cols].values
    y_test = test_df["target"].values

    model = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        reg_alpha=0.0,
        reg_lambda=1.0,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    y_pred = model.predict(X_test)
    metrics = evaluate_metrics(y_test, y_pred)

    model_filename = f"{target_name}_model.pkl"
    model_path = MODELS_DIR / model_filename
    joblib.dump(
        {
            "model": model,
            "lag": LAG,
            "feature_cols": feature_cols,
            "target_name": target_name,
        },
        model_path,
    )

    return ModelResult(
        target=target_name,
        metrics=metrics,
        model_path=str(model_path),
    )


def save_metrics(results: list[ModelResult]) -> Path:
    rows = []
    for result in results:
        row = {"target": result.target, "model_path": result.model_path}
        row.update(result.metrics)
        rows.append(row)

    metrics_df = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / "training_metrics.csv"
    metrics_df.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    ensure_directories()
    df = load_dataset(DATASET_PATH)

    results = []
    for target in ["pm25", "temperature", "humidity"]:
        results.append(train_one_model(target, df))

    metrics_file = save_metrics(results)

    print("\nTraining selesai.")
    print(f"Dataset bersih: {len(df)} baris")
    print(f"File metrik: {metrics_file}")
    for r in results:
        print(f"\nTarget: {r.target}")
        print(f"Model: {r.model_path}")
        for k, v in r.metrics.items():
            print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()