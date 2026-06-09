"""Export the Optuna-tuned XGBoost model and preprocessing artifacts for Streamlit."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import mlflow
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from carbon.features import CATEGORICAL_MAPS, NUMERIC_COLS  # noqa: E402
from carbon.paths import (  # noqa: E402
    COLUMNS_FILE,
    META_FILE,
    MLFLOW_DB,
    MODEL_DIR,
    MODEL_FILE,
    OPTUNA_DB,
    PROCESSED_CSV,
    SCALER_FILE,
    TRACKING_DIR,
)


def load_best_params():
    study = optuna.load_study(
        study_name="carbon_emission_xgb",
        storage=f"sqlite:///{OPTUNA_DB.as_posix()}",
    )
    return study.best_params, study.best_trial.number, study.best_value


def try_load_mlflow_model():
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.as_posix()}")
    model_uri = "runs:/b050c6cf407e4fd1be39e8f4dbca8185/xgboost_model"
    try:
        return mlflow.xgboost.load_model(model_uri)
    except Exception:
        return None


def train_and_export():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    TRACKING_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PROCESSED_CSV)
    feature_cols = [c for c in df.columns if c not in ("CarbonEmission", "Unnamed: 0")]
    X = df[feature_cols]
    y = df["CarbonEmission"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    best_params, best_trial, best_cv_r2 = load_best_params()
    model = try_load_mlflow_model()
    if model is None:
        model = xgb.XGBRegressor(
            **best_params,
            random_state=42,
            objective="reg:squarederror",
        )
        model.fit(X_train, y_train)
        source = "retrained"
    else:
        source = "mlflow"

    train_r2 = r2_score(y_train, model.predict(X_train))
    test_r2 = r2_score(y_test, model.predict(X_test))

    raw = pd.read_csv(
        "hf://datasets/We-Bears/Individual-Carbon-Footprint-Calculation/Carbon Emission.csv"
    )
    scaler = StandardScaler()
    scaler.fit(raw[NUMERIC_COLS])

    metadata = {
        "model_type": "xgboost",
        "best_trial": best_trial,
        "best_cv_r2": best_cv_r2,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "best_params": best_params,
        "source": source,
        "numeric_cols": NUMERIC_COLS,
        "feature_cols": feature_cols,
        "categorical_maps": CATEGORICAL_MAPS,
    }

    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(feature_cols, COLUMNS_FILE)
    META_FILE.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Exported model ({source})")
    print(f"  Best trial : {best_trial}")
    print(f"  CV R2      : {best_cv_r2:.4f}")
    print(f"  Train R2   : {train_r2:.4f}")
    print(f"  Test R2    : {test_r2:.4f}")
    print(f"  Features   : {len(feature_cols)}")
    print(f"  Output dir : {MODEL_DIR}")


if __name__ == "__main__":
    train_and_export()
