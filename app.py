import streamlit as st
import pandas as pd
import joblib
import numpy as np
import ollama

# Load the saved model and preprocessing objects
model = joblib.load('carbon_footprint_model.pkl')
scaler = joblib.load('carbon_scaler.pkl')
expected_columns = joblib.load('carbon_columns.pkl')

def get_ai_insights(user_profile, prediction):
    prompt = f"""
You are a climate impact expert.

User lifestyle data:
{user_profile}

Predicted annual carbon footprint: {prediction:.2f} kg CO2.

Give:
1. Main reasons for this emission level
2. 3 specific actions to reduce it
3. One blunt truth about their habits

Keep it short, clear, and practical.
"""
    response = ollama.chat(
        model="gemini-3-flash-preview:cloud",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]

st.title("🌍 Carbon Footprint Calculator")
st.markdown("Calculate your personal carbon emissions based on your lifestyle and habits.")

tab1, tab2, tab3 = st.tabs(["Personal Info", "Lifestyle", "Transportation & Energy"])

with tab1:
    sex = st.selectbox("Gender", ["Male", "Female"])
    body_type = st.selectbox("Body Type", ["Normal", "Obese", "Overweight", "Underweight"])
    monthly_grocery = st.number_input("Monthly Grocery Bill ($)", 0, 1000, 200)

with tab2:
    diet = st.selectbox("Diet Type", ["Omnivore", "Pescatarian", "Vegan", "Vegetarian"])
    shower_frequency = st.selectbox("Shower Frequency", ["Daily", "Less Frequently", "More Frequently", "Twice a Day"])
    social_activity = st.selectbox("Social Activity", ["Never", "Often", "Sometimes"])
    tv_pc_hours = st.slider("TV/PC Daily Hours", 0, 24, 4)
    internet_hours = st.slider("Internet Daily Hours", 0, 24, 3)
    new_clothes = st.number_input("New Clothes Monthly", 0, 50, 2)
    waste_bags = st.number_input("Waste Bags Weekly", 0, 20, 3)
    waste_bag_size = st.selectbox("Waste Bag Size", ["Small", "Medium", "Large", "Extra Large"])
    recycling = st.multiselect("Recycling", ["Glass", "Metal", "Paper", "Plastic"])
    cooking_methods = st.multiselect("Cooking Methods", ["Airfryer", "Grill", "Microwave", "Oven", "Stove"])

with tab3:
    transport = st.selectbox("Primary Transport", ["Private", "Public", "Walk/Bicycle"])
    vehicle_type = st.selectbox("Vehicle Type", ["None", "Electric", "Hybrid", "LPG", "Petrol", "Unknown"])
    vehicle_distance = st.number_input("Vehicle Monthly Distance (Km)", 0, 5000, 500)
    air_travel = st.selectbox("Air Travel Frequency", ["Never", "Rarely", "Frequently", "Very Frequently"])
    heating_source = st.selectbox("Heating Source", ["Coal", "Electricity", "Natural Gas", "Wood"])
    energy_efficiency = st.selectbox("Energy Efficiency", ["No", "Sometimes", "Yes"])

if st.button("Calculate Carbon Footprint", type="primary", use_container_width=True):

    input_data = {col: 0 for col in expected_columns}

    input_data['Monthly Grocery Bill'] = monthly_grocery
    input_data['Vehicle Monthly Distance Km'] = vehicle_distance
    input_data['Waste Bag Weekly Count'] = waste_bags
    input_data['How Long TV PC Daily Hour'] = tv_pc_hours
    input_data['How Many New Clothes Monthly'] = new_clothes
    input_data['How Long Internet Daily Hour'] = internet_hours
    input_data['Sex'] = 1 if sex == "Female" else 0

    input_df = pd.DataFrame([input_data])
    input_df = input_df[expected_columns]

    numerical_cols = [
        'Monthly Grocery Bill',
        'Vehicle Monthly Distance Km',
        'Waste Bag Weekly Count',
        'How Long TV PC Daily Hour',
        'How Many New Clothes Monthly',
        'How Long Internet Daily Hour'
    ]
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    prediction = model.predict(input_df)[0]

    st.success(f"### 🌱 Annual Carbon Footprint: {prediction:,.2f} kg CO₂")

    st.metric("Daily Avg", f"{prediction/365:.2f} kg")
    st.metric("Monthly Avg", f"{prediction/12:.2f} kg")

    st.subheader("🤖 AI Expert Insights (Powered by Ollama)")

    user_profile = {
        "Gender": sex,
        "Body Type": body_type,
        "Diet": diet,
        "Transport": transport,
        "Vehicle Type": vehicle_type,
        "Monthly Distance": vehicle_distance,
        "Air Travel": air_travel,
        "Heating Source": heating_source,
        "Energy Efficiency": energy_efficiency,
        "Recycling": recycling,
        "Cooking Methods": cooking_methods
    }

    with st.spinner("AI is analyzing your lifestyle..."):
        ai_insights = get_ai_insights(user_profile, prediction)

    st.write(ai_insights)

    st.caption("⚠ This is an estimate. Real emissions depend on many real-world factors.")
