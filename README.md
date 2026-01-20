# 🌱 Carbon Footprint Calculator

A machine learning-powered web application that predicts individual carbon emissions based on lifestyle, transportation, and consumption habits.

## 📖 Overview

This project analyzes personal carbon footprints by considering multiple factors including diet, transportation, energy usage, and waste management. Built with Linear Regression and deployed via Streamlit, it provides real-time emission estimates (kg CO₂/year) and offers actionable recommendations for a sustainable lifestyle.

## 🚀 Features

- **Real-Time Predictions**: Instant carbon footprint calculations based on user inputs.
- **Interactive Interface**: User-friendly, tab-based Streamlit application.
- **Smart Recommendations**: Personalized tips to reduce environmental impact.
- **Robust Preprocessing**: Automated scaling, multi-label handling, and statistical feature selection.
- **Data Visualization**: Comparative metrics against average carbon footprints.

## 🛠️ Quick Start

### Installation

Clone the repository and install the required dependencies.

#### Clone the repository
```
git clone https://github.com/deepseek23/Carbon-Footprint-Prediction.git
cd carbon-footprint-calculator
```

#### Install dependencies
```
pip install -r requirements.txt
```

### Initialize Model

Note: You must train the model once to generate the required .pkl files before running the app.

#### Open the notebook:
```
jupyter notebook carbon_footprint.ipynb
```

Run all cells to process the data and train the model.

Verify that `carbon_footprint_model.pkl`, `carbon_scaler.pkl`, and `carbon_columns.pkl` have been created.

### Run Application

Launch the web interface:
```
streamlit run app.py
```

The application will open automatically in your browser at http://localhost:8501.

## 📊 Dataset & Model

- **Data Source**: Individual Carbon Footprint Calculation Dataset (10,000+ records).
- **Algorithm**: Linear Regression.
- **Performance**: ~0.78 R² Score on test data.
- **Key Predictors**: Air travel frequency, vehicle type, and body type are the most significant drivers of emissions.

## 🤝 Contributing

Contributions are welcome!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/NewFeature`)
3. Commit your Changes (`git commit -m 'Add NewFeature'`)
4. Push to the Branch (`git push origin feature/NewFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

Made with 🌱 for a greener planet