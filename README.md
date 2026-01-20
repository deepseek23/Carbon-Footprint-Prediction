# Carbon Footprint Calculator

A machine learning-powered web application that predicts individual carbon emissions based on lifestyle, transportation, and consumption habits. Built with Linear Regression and deployed using Streamlit for real-time predictions.

## 🌍 Overview

This project analyzes personal carbon footprint by considering multiple factors including:
- **Personal Information**: Gender, body type, grocery spending
- **Lifestyle Habits**: Diet type, shower frequency, social activities, entertainment hours
- **Waste Management**: Waste generation, recycling habits, cooking methods
- **Transportation**: Vehicle type, distance traveled, air travel frequency
- **Energy Consumption**: Heating sources, energy efficiency practices

The model provides personalized carbon emission estimates in kg CO₂ per year and offers actionable recommendations to reduce environmental impact.

## 🚀 Features

- **Interactive Web Interface**: User-friendly Streamlit application with organized tabs
- **Real-Time Predictions**: Instant carbon footprint calculations based on user inputs
- **Data Preprocessing**: Automated feature encoding, scaling, and multi-label handling
- **Statistical Feature Selection**: Chi-square test to identify significant predictors
- **Personalized Recommendations**: Smart insights based on user behavior
- **Comprehensive Metrics**: Daily, monthly, and annual emission breakdowns
- **Comparison Analysis**: Benchmark against average carbon footprint

## 📊 Dataset

The project uses the [Individual Carbon Footprint Calculation Dataset](https://huggingface.co/datasets/We-Bears/Individual-Carbon-Footprint-Calculation) from Hugging Face, containing 10,000+ records with the following features:

**Numerical Features:**
- Monthly Grocery Bill
- Vehicle Monthly Distance (Km)
- Waste Bag Weekly Count
- How Long TV/PC Daily (Hours)
- How Many New Clothes Monthly
- How Long Internet Daily (Hours)

**Categorical Features:**
- Sex (Male/Female)
- Body Type (Normal, Obese, Overweight, Underweight)
- Diet (Omnivore, Pescatarian, Vegan, Vegetarian)
- How Often Shower (Daily, Less/More Frequently, Twice a Day)
- Heating Energy Source (Coal, Electricity, Natural Gas, Wood)
- Transport (Private, Public, Walk/Bicycle)
- Social Activity (Never, Often, Sometimes)
- Frequency of Air Travel (Never, Rarely, Frequently, Very Frequently)
- Waste Bag Size (Small, Medium, Large, Extra Large)
- Vehicle Type (None, Electric, Hybrid, LPG, Petrol)
- Energy Efficiency (No, Sometimes, Yes)

**Multi-Label Features:**
- Recycling (Glass, Metal, Paper, Plastic)
- Cooking Methods (Airfryer, Grill, Microwave, Oven, Stove)

**Target Variable:**
- CarbonEmission (kg CO₂ per year)

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/deepseek23/carbon-footprint-calculator.git
   cd carbon-footprint-calculator
   ```

2. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify the model files exist:**
   Ensure the following files are in the project directory:
   - `carbon_footprint_model.pkl`
   - `carbon_scaler.pkl`
   - `carbon_columns.pkl`

## 📦 Dependencies

Create a `requirements.txt` file with the following packages:

```
numpy==1.24.3
pandas==2.0.3
seaborn==0.12.2
matplotlib==3.7.2
scikit-learn==1.3.0
scipy==1.11.1
joblib==1.3.2
streamlit==1.28.1
```

## 🎯 Usage

### Running the Jupyter Notebook

1. **Open the notebook:**
   ```bash
   jupyter notebook carbon_footprint.ipynb
   ```

2. **Execute cells sequentially** to perform:
   - Exploratory Data Analysis (EDA)
   - Data cleaning and preprocessing
   - Feature engineering (multi-label encoding, one-hot encoding)
   - Feature scaling with StandardScaler
   - Chi-square statistical testing for feature selection
   - Model training with Linear Regression
   - Model evaluation and visualization
   - Model serialization using joblib

3. **Model files generated:**
   - `carbon_footprint_model.pkl` - Trained Linear Regression model
   - `carbon_scaler.pkl` - StandardScaler for numerical features
   - `carbon_columns.pkl` - Feature column names for consistency

### Running the Streamlit App

After training the model and generating the pickle files:

1. **Start the application:**
   ```bash
   streamlit run app.py
   ```

2. **Access the web interface:**
   - The app will automatically open in your default browser
   - Default URL: `http://localhost:8501`

3. **Use the application:**
   - Navigate through three tabs: Personal Info, Lifestyle, and Transportation & Energy
   - Fill in your information across all categories
   - Click "Calculate Carbon Footprint" to get your prediction
   - Review personalized recommendations and insights

## 🤖 Model Development

### Data Preprocessing Pipeline

1. **Handling Missing Values:**
   - Vehicle Type: Filled with 'unknown' category
   
2. **Duplicate Removal:**
   - Removed duplicate records for data quality

3. **Encoding Categorical Variables:**
   - **Label Encoding**: Sex (Male: 0, Female: 1)
   - **One-Hot Encoding**: Body Type, Diet, Shower Frequency, Heating Source, Transport, Social Activity, Air Travel, Waste Bag Size, Vehicle Type, Energy Efficiency
   - **Multi-Label Binarization**: Recycling habits, Cooking methods

4. **Feature Scaling:**
   - StandardScaler applied to numerical features for normalization

5. **Feature Selection:**
   - Chi-square test with α = 0.05 significance level
   - Dropped 13 non-significant features (p-value ≥ 0.05)
   - Final dataset: 33 features from original 47

### Model Architecture

- **Algorithm**: Linear Regression
- **Train-Test Split**: 80-20 ratio with random_state=42
- **Input Features**: 33 selected features after preprocessing
- **Output**: Continuous value (Carbon Emission in kg CO₂/year)

### Model Performance

- **R² Score**: ~0.78 on test set
- **Adjusted R²**: Accounts for number of predictors
- **Residual Analysis**: Validated model assumptions
- **Learning Curves**: Assessed model generalization

### Feature Importance (Top Predictors)

Based on chi-square statistical testing:
1. Frequency of Air Travel (Very Frequently)
2. Vehicle Type (Unknown/Petrol/LPG)
3. Frequency of Air Travel (Never)
4. Transport Mode (Walk/Bicycle)
5. Body Type (Obese)
6. Gender
7. Heating Energy Source
8. Waste Management habits

## 📈 Model Evaluation Metrics

The notebook includes comprehensive evaluation:

- **R² Score**: Measures proportion of variance explained
- **Adjusted R² Score**: Accounts for model complexity
- **Residual Plots**: Visualizes prediction errors
- **Learning Curves**: Shows training vs validation performance
- **Feature Correlation Heatmap**: Identifies multicollinearity

## 🎨 Application Features

### User Interface Components

1. **Tab 1 - Personal Information:**
   - Gender selection
   - Body type classification
   - Monthly grocery spending

2. **Tab 2 - Lifestyle:**
   - Diet preferences
   - Shower frequency
   - Entertainment hours (TV/PC, Internet)
   - Shopping habits
   - Waste generation
   - Recycling practices
   - Cooking methods

3. **Tab 3 - Transportation & Energy:**
   - Primary transport mode
   - Vehicle specifications
   - Travel distance
   - Air travel frequency
   - Heating energy source
   - Energy efficiency rating

### Output Display

- **Primary Metric**: Annual carbon footprint in kg CO₂
- **Breakdown Metrics**: Daily and monthly emissions
- **Comparison**: Above/below average benchmark
- **Recommendations**: Up to 6 personalized suggestions
- **Visual Indicators**: Color-coded metrics and icons

## 🔬 Technical Implementation

### Data Processing Flow

```
Raw Data → Missing Value Treatment → Duplicate Removal → 
Feature Encoding → Scaling → Feature Selection → 
Train-Test Split → Model Training → Evaluation → Serialization
```

### Prediction Pipeline

```
User Input → Feature Initialization → Encoding → 
DataFrame Creation → Column Alignment → Scaling → 
Model Prediction → Result Display → Recommendations
```

## 📁 Project Structure

```
carbon-footprint-calculator/
│
├── carbon_footprint.ipynb      # Main analysis and model training notebook
├── app.py                       # Streamlit web application
├── carbon_footprint_model.pkl  # Trained Linear Regression model
├── carbon_scaler.pkl           # StandardScaler object
├── carbon_columns.pkl          # Feature column names
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── .gitignore                  # Git ignore file
```

## 💡 Key Insights

From the exploratory data analysis and model training:

1. **Air Travel Impact**: Most significant predictor of carbon footprint
2. **Vehicle Type Matters**: Unknown/Petrol vehicles correlate with higher emissions
3. **Active Transport**: Walking/cycling significantly reduces footprint
4. **Body Type Correlation**: Obesity associated with higher consumption patterns
5. **Waste Management**: Recycling practices show measurable impact
6. **Energy Efficiency**: Mixed results, suggesting behavioral variation

## 🌟 Future Enhancements

- [ ] Add more sophisticated models (Random Forest, XGBoost, Neural Networks)
- [ ] Implement feature importance visualization in the app
- [ ] Add time-series analysis for emission trends
- [ ] Include carbon offset recommendations
- [ ] Add comparison with country-specific averages
- [ ] Implement user authentication and history tracking
- [ ] Add data export functionality
- [ ] Include more granular location-based factors
- [ ] Mobile-responsive design improvements
- [ ] Multi-language support

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset: [We-Bears Individual Carbon Footprint Calculation](https://huggingface.co/datasets/We-Bears/Individual-Carbon-Footprint-Calculation)
- Streamlit for the amazing web framework
- Scikit-learn for machine learning tools
- The open-source community

## 📞 Contact

For questions, suggestions, or collaboration opportunities, please open an issue on GitHub.

---

**Made with 🌱 for a greener planet**
