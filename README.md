# Carbon Footprint Calculator

A machine learning–powered Streamlit application that estimates an individual’s annual carbon footprint (kg CO₂/year) from lifestyle, transportation, and consumption patterns.

## Overview

This project predicts personal carbon emissions by combining multiple signals (e.g., diet, travel behavior, household energy usage, and waste management). The model is trained in the accompanying notebook and deployed as an interactive web app for real-time inference and actionable reduction recommendations.

## Key Features

- Real-time carbon footprint prediction from user inputs
- Streamlit-based interactive UI
- Personalized reduction recommendations
- AI-powered insights via Ollama (local LLM)
- Reproducible preprocessing pipeline (encoding + scaling)
- Visualization and comparison against benchmark/average footprints

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

The application expects trained artifacts to exist locally before it can run.

1. Open the notebook:
	```bash
	jupyter notebook carbon_footprint.ipynb
	```

2. Run all cells to preprocess data and train the model.

3. Confirm that these files are created:

- `carbon_footprint_model.pkl`
- `carbon_scaler.pkl`
- `carbon_columns.pkl`

## Run the Application

Start the Streamlit app:

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## AI Insights (Ollama)

This project can optionally use **Ollama** to generate AI-driven insights (e.g., concise explanations and actionable suggestions based on the user’s inputs and predicted footprint). Ollama runs locally on your machine, keeping the inference workflow self-contained.

If you want to enable AI insights:

1. Install Ollama: https://ollama.com
2. Pull a model (example):
	```bash
	ollama pull llama3
	```
3. Ensure Ollama is running before launching the Streamlit app.

## Dataset and Model

- **Dataset**: Individual Carbon Footprint Calculation Dataset (10,000+ records)
- **Model**: Linear Regression
- **Test performance**: ~0.78 $R^2$
- **Notable drivers**: air travel frequency, vehicle type, and body type

### Model Description

The system uses a **Linear Regression** model to estimate annual emissions from mixed feature types (numeric + categorical). For consistent training and inference, the pipeline encodes categorical features into numeric representations and standardizes numeric inputs (e.g., using `StandardScaler`). The final feature set and column ordering are preserved to ensure that the Streamlit app applies transformations in the same way as the training notebook.

### Deployment Artifacts

The notebook exports the following artifacts for deployment:

- `carbon_footprint_model.pkl` — trained regression model
- `carbon_scaler.pkl` — fitted scaler for numeric features
- `carbon_columns.pkl` — ordered feature/column list used during training

## Contributing

Contributions are welcome.

1. Fork the repository
2. Create a new branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m "Add my feature"`
4. Push the branch: `git push origin feature/my-feature`
5. Open a Pull Request

## License

This project is licensed under the MIT License. See `LICENSE` for details.