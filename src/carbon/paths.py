"""Project path constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

APP_DIR = PROJECT_ROOT / "app"
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "artifacts" / "model"
TRACKING_DIR = PROJECT_ROOT / "tracking"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

def _tracking_file(name: str) -> Path:
    """Prefer tracking/; fall back to project root for legacy layouts."""
    preferred = TRACKING_DIR / name
    if preferred.exists():
        return preferred
    legacy = PROJECT_ROOT / name
    return legacy if legacy.exists() else preferred


OPTUNA_DB = _tracking_file("optuna_study.db")
MLFLOW_DB = _tracking_file("mlflow.db")
MLARTIFACTS_DIR = TRACKING_DIR / "mlartifacts"
MLRUNS_DIR = TRACKING_DIR / "mlruns"

MODEL_FILE = MODEL_DIR / "carbon_footprint_model.pkl"
SCALER_FILE = MODEL_DIR / "carbon_scaler.pkl"
COLUMNS_FILE = MODEL_DIR / "carbon_columns.pkl"
META_FILE = MODEL_DIR / "carbon_model_meta.json"
PROCESSED_CSV = PROCESSED_DATA_DIR / "new_carbon.csv"
