"""Evaluate Optuna trials on hold-out data and log metrics to MLflow."""

import sys
from pathlib import Path

import mlflow
import optuna
import pandas as pd
import xgboost as xgb
from mlflow.tracking import MlflowClient
from optuna.trial import TrialState
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from carbon.paths import MLFLOW_DB, OPTUNA_DB, PROCESSED_CSV  # noqa: E402

mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.as_posix()}")
exp_name = "CarbonEmission_Optuna_Trials"
exp_id = mlflow.get_experiment_by_name(exp_name).experiment_id
client = MlflowClient()

study = optuna.load_study(
    study_name="carbon_emission_xgb",
    storage=f"sqlite:///{OPTUNA_DB.as_posix()}",
)

df = pd.read_csv(PROCESSED_CSV)
X = df.drop(columns=["CarbonEmission", "Unnamed: 0"], errors="ignore")
y = df["CarbonEmission"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

evaluated = 0
for trial in study.trials:
    if trial.state != TrialState.COMPLETE or trial.value is None:
        continue
    runs = client.search_runs(
        [exp_id],
        filter_string=f"tags.optuna_trial_number = '{trial.number}'",
        max_results=1,
    )
    if not runs or "test_r2" in runs[0].data.metrics:
        continue

    params = trial.params.copy()
    params.update(random_state=42, objective="reg:squarederror", tree_method="hist")
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    with mlflow.start_run(run_id=runs[0].info.run_id):
        mlflow.log_metrics(
            {
                "train_r2": r2_score(y_train, train_pred),
                "test_r2": r2_score(y_test, test_pred),
                "train_rmse": mean_squared_error(y_train, train_pred) ** 0.5,
                "test_rmse": mean_squared_error(y_test, test_pred) ** 0.5,
                "train_mae": mean_absolute_error(y_train, train_pred),
                "test_mae": mean_absolute_error(y_test, test_pred),
            }
        )
    evaluated += 1
    if evaluated % 10 == 0:
        print(f"Evaluated {evaluated} trials...")

print(f"Done. Evaluated {evaluated} trials.")
