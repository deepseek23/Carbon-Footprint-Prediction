# Carbon Footprint Calculator

A machine learning–powered Streamlit application that estimates an individual’s annual carbon footprint (kg CO₂/year) from lifestyle, transportation, and consumption patterns.

## Overview

This project predicts personal carbon emissions by combining multiple signals (e.g., diet, travel behavior, household energy usage, and waste management). The model is trained in the accompanying notebook and deployed as an interactive web app for real-time inference and actionable reduction recommendations.

## Project Structure

```
Carbon footprint/
├── app/
│   └── streamlit_app.py      # Streamlit UI
├── artifacts/
│   └── model/                # Deployed model files (.pkl + meta JSON)
├── data/
│   └── processed/            # Engineered training data (new_carbon.csv)
├── notebooks/
│   └── carbon_footprint.ipynb
├── scripts/
│   ├── export_model.py       # Export best model from MLflow → artifacts/
│   └── eval_trials.py        # Log train/test metrics for Optuna trials
├── src/
│   └── carbon/               # Shared paths, features, AI insights
│       ├── paths.py
│       ├── features.py
│       └── insights.py
├── tracking/                 # MLflow + Optuna (local, gitignored)
├── requirements.txt
└── README.md
```

## Key Features

- Real-time carbon footprint prediction from user inputs
- Streamlit-based interactive UI
- Personalized reduction recommendations
- AI-powered insights via Ollama (local LLM)
- Reproducible preprocessing pipeline (encoding + scaling)
- Visualization and comparison against benchmark/average footprints
- **XGBoost + Optuna** hyperparameter tuning (100 trials, ~0.98 test $R^2$)
- **MLflow** experiment tracking and trial comparison UI

## Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook (for training)
- Streamlit (for the web app)
- Ollama (optional, for AI insights)

### Installation

1. Clone the repository:
	```bash
	git clone https://github.com/deepseek23/Carbon-Footprint-Prediction.git
	cd carbon-footprint-calculator
	```

2. Install dependencies:
	```bash
	pip install -r requirements.txt
	```

## Train the Model (One-Time)

The application expects trained artifacts in `artifacts/model/` before it can run.

1. Open the notebook:
	```bash
	jupyter notebook notebooks/carbon_footprint.ipynb
	```

2. Run all cells to preprocess data and train the model.

3. Run the Optuna / MLflow cells, then export:
	```bash
	python scripts/export_model.py
	```

4. Confirm these files exist under `artifacts/model/`:

- `carbon_footprint_model.pkl` — Optuna-tuned XGBoost model
- `carbon_scaler.pkl` — fitted scaler for numeric features
- `carbon_columns.pkl` — ordered feature column list
- `carbon_model_meta.json` — best trial, params, and test $R^2$

## Run the Application

```bash
streamlit run app/streamlit_app.py
```

Then open http://localhost:8501 in your browser.

## AI Insights (Ollama)

This project can optionally use **Ollama** to generate structured coaching insights (main drivers, actions, reality check) from the user’s inputs and predicted footprint.

1. Install Ollama: https://ollama.com
2. Pull a model (example):
	```bash
	ollama pull minimax-m2.5:cloud
	```
3. Ensure Ollama is running before launching the Streamlit app.

## Dataset and Model

- **Dataset**: Individual Carbon Footprint Calculation Dataset (10,000+ records)
- **Baseline model**: Linear Regression (~0.92 test $R^2$ on the filtered feature set)
- **Current best model**: XGBoost tuned with Optuna
- **Best test performance**: **0.9773** $R^2$ (train $R^2$: 0.9840)
- **Notable drivers**: air travel frequency, vehicle type, and body type

### Model Description

The notebook first trains a **Linear Regression** baseline on encoded categorical and scaled numeric features. It then upgrades to an **XGBoost** regressor with automated hyperparameter search via **Optuna** and experiment tracking via **MLflow**.

For consistent training and inference, the pipeline encodes categorical features into numeric representations and standardizes numeric inputs (e.g., using `StandardScaler`). The preprocessed dataset is saved as `data/processed/new_carbon.csv` for reproducible Optuna runs.

### Hyperparameter Tuning (Optuna + MLflow)

1. **Optuna study** (`carbon_emission_xgb`) — 100 trials in `tracking/optuna_study.db`
2. **MLflow import** — trials synced to `CarbonEmission_Optuna_Trials` in `tracking/mlflow.db`
3. **Best-model run** — winning trial logged to `CarbonEmission_XGBoost_Optuna`
4. **MLflow UI** — compare trials and sort by `cv_r2` or `test_r2`

**Best trial (#91)**

| Metric | Score |
|--------|-------|
| CV $R^2$ (5-fold, Optuna objective) | **0.9750** |
| Train $R^2$ | **0.9840** |
| Test $R^2$ | **0.9773** |

**Best hyperparameters**

| Parameter | Value |
|-----------|-------|
| `n_estimators` | 833 |
| `max_depth` | 3 |
| `learning_rate` | 0.0836 |
| `subsample` | 0.6826 |
| `colsample_bytree` | 0.8032 |
| `gamma` | 8.4987 |
| `min_child_weight` | 4 |
| `reg_alpha` | 0.0644 |
| `reg_lambda` | 5.8889 |

To open the MLflow dashboard:

```bash
mlflow ui --backend-store-uri sqlite:///tracking/mlflow.db --default-artifact-root tracking/mlartifacts
```

Then visit http://127.0.0.1:5000.

### Deployment Artifacts

Run `python scripts/export_model.py` after Optuna training to pull the best model from MLflow:

| File | Description |
|------|-------------|
| `artifacts/model/carbon_footprint_model.pkl` | Optuna-tuned XGBoost (~0.977 test $R^2$) |
| `artifacts/model/carbon_scaler.pkl` | Fitted scaler |
| `artifacts/model/carbon_columns.pkl` | Feature column order |
| `artifacts/model/carbon_model_meta.json` | Trial #, hyperparameters, metrics |
| `data/processed/new_carbon.csv` | Preprocessed training data |

## Contributing

Contributions are welcome.

1. Fork the repository
2. Create a new branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m "Add my feature"`
4. Push the branch: `git push origin feature/my-feature`
5. Open a Pull Request

## License

This project is licensed under the MIT License. See `LICENSE` for details.
