import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the saved model and preprocessing objects
model = joblib.load('carbon_footprint_model.pkl')
scaler = joblib.load('carbon_scaler.pkl')
expected_columns = joblib.load('carbon_columns.pkl')

# Title and description
st.title("🌍 Carbon Footprint Calculator")
st.markdown("Calculate your personal carbon emissions based on your lifestyle and habits.")

# Create tabs for better organization
tab1, tab2, tab3 = st.tabs(["Personal Info", "Lifestyle", "Transportation & Energy"])

with tab1:
    st.subheader("Personal Information")
    col1, col2 = st.columns(2)
    
    with col1:
        sex = st.selectbox("Gender", ["Male", "Female"])
        body_type = st.selectbox("Body Type", ["Normal", "Obese", "Overweight", "Underweight"])
    
    with col2:
        monthly_grocery = st.number_input("Monthly Grocery Bill ($)", min_value=0, max_value=1000, value=200, step=10)
        
with tab2:
    st.subheader("Lifestyle Habits")
    col1, col2 = st.columns(2)
    
    with col1:
        diet = st.selectbox("Diet Type", ["Omnivore", "Pescatarian", "Vegan", "Vegetarian"])
        shower_frequency = st.selectbox("How Often Do You Shower?", 
                                       ["Daily", "Less Frequently", "More Frequently", "Twice a Day"])
        social_activity = st.selectbox("Social Activity Level", 
                                      ["Never", "Often", "Sometimes"])
    
    with col2:
        tv_pc_hours = st.slider("TV/PC Daily Hours", 0, 24, 4)
        internet_hours = st.slider("Internet Daily Hours", 0, 24, 3)
        new_clothes = st.number_input("New Clothes Monthly", min_value=0, max_value=50, value=2, step=1)
        waste_bags = st.number_input("Waste Bag Weekly Count", min_value=0, max_value=20, value=3, step=1)
        
    st.subheader("Waste Management")
    col3, col4 = st.columns(2)
    
    with col3:
        waste_bag_size = st.selectbox("Waste Bag Size", ["Small", "Medium", "Large", "Extra Large"])
        recycling = st.multiselect("What Do You Recycle?", 
                                   ["Glass", "Metal", "Paper", "Plastic"])
    
    with col4:
        cooking_methods = st.multiselect("Cooking Methods You Use", 
                                        ["Airfryer", "Grill", "Microwave", "Oven", "Stove"])

with tab3:
    st.subheader("Transportation")
    col1, col2 = st.columns(2)
    
    with col1:
        transport = st.selectbox("Primary Transport", ["Private", "Public", "Walk/Bicycle"])
        vehicle_type = st.selectbox("Vehicle Type", 
                                   ["None", "Electric", "Hybrid", "LPG", "Petrol", "Unknown"])
        vehicle_distance = st.number_input("Vehicle Monthly Distance (Km)", 
                                          min_value=0, max_value=5000, value=500, step=50)
    
    with col2:
        air_travel = st.selectbox("Frequency of Air Travel", 
                                 ["Never", "Rarely", "Frequently", "Very Frequently"])
    
    st.subheader("Energy")
    col3, col4 = st.columns(2)
    
    with col3:
        heating_source = st.selectbox("Heating Energy Source", 
                                     ["Coal", "Electricity", "Natural Gas", "Wood"])
    
    with col4:
        energy_efficiency = st.selectbox("Energy Efficiency", ["No", "Sometimes", "Yes"])

# Prediction button
if st.button("Calculate Carbon Footprint", type="primary", use_container_width=True):
    # Create input dictionary with all features initialized to 0
    input_data = {}
    
    # Initialize all expected columns to 0
    for col in expected_columns:
        input_data[col] = 0
    
    # Set numeric features (will be scaled later)
    input_data['Monthly Grocery Bill'] = monthly_grocery
    input_data['Vehicle Monthly Distance Km'] = vehicle_distance
    input_data['Waste Bag Weekly Count'] = waste_bags
    input_data['How Long TV PC Daily Hour'] = tv_pc_hours
    input_data['How Many New Clothes Monthly'] = new_clothes
    input_data['How Long Internet Daily Hour'] = internet_hours
    
    # Set Sex (0 for male, 1 for female)
    input_data['Sex'] = 1 if sex == "Female" else 0
    
    # Set Body Type (one-hot encoded, drop_first=True, so 'normal' is baseline)
    if body_type == "Obese":
        input_data['Body Type_obese'] = 1
    elif body_type == "Overweight":
        input_data['Body Type_overweight'] = 1
    elif body_type == "Underweight":
        input_data['Body Type_underweight'] = 1
    
    # Set Diet (one-hot encoded, drop_first=True, so 'omnivore' is baseline)
    if diet == "Pescatarian":
        input_data['Diet_pescatarian'] = 1
    elif diet == "Vegan":
        input_data['Diet_vegan'] = 1
    elif diet == "Vegetarian":
        input_data['Diet_vegetarian'] = 1
    
    # Set Shower Frequency
    if shower_frequency == "Less Frequently":
        input_data['How Often Shower_less frequently'] = 1
    elif shower_frequency == "More Frequently":
        input_data['How Often Shower_more frequently'] = 1
    elif shower_frequency == "Twice a Day":
        input_data['How Often Shower_twice a day'] = 1
    
    # Set Heating Source
    if heating_source == "Electricity":
        input_data['Heating Energy Source_electricity'] = 1
    elif heating_source == "Natural Gas":
        input_data['Heating Energy Source_natural gas'] = 1
    elif heating_source == "Wood":
        input_data['Heating Energy Source_wood'] = 1
    
    # Set Transport
    if transport == "Public":
        input_data['Transport_public'] = 1
    elif transport == "Walk/Bicycle":
        input_data['Transport_walk/bicycle'] = 1
    
    # Set Social Activity
    if social_activity == "Often":
        input_data['Social Activity_often'] = 1
    elif social_activity == "Sometimes":
        input_data['Social Activity_sometimes'] = 1
    
    # Set Air Travel
    if air_travel == "Never":
        input_data['Frequency of Traveling by Air_never'] = 1
    elif air_travel == "Rarely":
        input_data['Frequency of Traveling by Air_rarely'] = 1
    elif air_travel == "Very Frequently":
        input_data['Frequency of Traveling by Air_very frequently'] = 1
    
    # Set Waste Bag Size
    if waste_bag_size == "Large":
        input_data['Waste Bag Size_large'] = 1
    elif waste_bag_size == "Medium":
        input_data['Waste Bag Size_medium'] = 1
    elif waste_bag_size == "Small":
        input_data['Waste Bag Size_small'] = 1
    
    # Set Vehicle Type
    if vehicle_type == "Electric":
        input_data['Vehicle Type_electric'] = 1
    elif vehicle_type == "Hybrid":
        input_data['Vehicle Type_hybrid'] = 1
    elif vehicle_type == "LPG":
        input_data['Vehicle Type_lpg'] = 1
    elif vehicle_type == "Petrol":
        input_data['Vehicle Type_petrol'] = 1
    elif vehicle_type == "Unknown":
        input_data['Vehicle Type_unknown'] = 1
    
    # Set Energy Efficiency
    if energy_efficiency == "Sometimes":
        input_data['Energy efficiency_Sometimes'] = 1
    elif energy_efficiency == "Yes":
        input_data['Energy efficiency_Yes'] = 1
    
    # Set Recycling (multi-label)
    if "Glass" in recycling:
        input_data['Glass'] = 1
    if "Metal" in recycling:
        input_data['Metal'] = 1
    if "Paper" in recycling:
        input_data['Paper'] = 1
    if "Plastic" in recycling:
        input_data['Plastic'] = 1
    
    # Set Cooking Methods (multi-label)
    if "Airfryer" in cooking_methods:
        input_data['Airfryer'] = 1
    if "Grill" in cooking_methods:
        input_data['Grill'] = 1
    if "Microwave" in cooking_methods:
        input_data['Microwave'] = 1
    if "Oven" in cooking_methods:
        input_data['Oven'] = 1
    if "Stove" in cooking_methods:
        input_data['Stove'] = 1
    
    # Create DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Ensure all expected columns are present in the correct order
    input_df = input_df[expected_columns]
    
    # Scale the numerical columns
    numerical_cols = ['Monthly Grocery Bill', 'Vehicle Monthly Distance Km', 
                     'Waste Bag Weekly Count', 'How Long TV PC Daily Hour', 
                     'How Many New Clothes Monthly', 'How Long Internet Daily Hour']
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    
    # Display result
    st.success(f"### 🌱 Your Estimated Annual Carbon Footprint: **{prediction:,.2f} kg CO₂**")
    
    # Provide context
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Daily Emissions", f"{prediction/365:.2f} kg", help="Average daily CO₂ emissions")
    with col2:
        st.metric("Monthly Emissions", f"{prediction/12:.2f} kg", help="Average monthly CO₂ emissions")
    with col3:
        # Average per capita is around 4000-5000 kg/year
        comparison = "Above Average" if prediction > 4500 else "Below Average"
        st.metric("Compared to Average", comparison, help="Average person: ~4500 kg/year")
    
    # Environmental impact insights
    st.markdown("---")
    st.subheader("💡 Insights & Recommendations")
    
    recommendations = []
    
    if vehicle_distance > 1000:
        recommendations.append("🚗 Consider using public transport or carpooling to reduce vehicle emissions.")
    
    if air_travel in ["Frequently", "Very Frequently"]:
        recommendations.append("✈️ Air travel has a significant carbon footprint. Consider video calls or train travel when possible.")
    
    if transport == "Private" and vehicle_type == "Petrol":
        recommendations.append("🔋 Switching to a hybrid or electric vehicle could significantly reduce your emissions.")
    
    if "Paper" not in recycling and "Plastic" not in recycling:
        recommendations.append("♻️ Start recycling paper and plastic to reduce waste-related emissions.")
    
    if energy_efficiency == "No":
        recommendations.append("💡 Invest in energy-efficient appliances and LED bulbs to reduce energy consumption.")
    
    if diet == "Omnivore":
        recommendations.append("🥗 Reducing meat consumption, even by a few days per week, can lower your carbon footprint.")
    
    if recommendations:
        for rec in recommendations:
            st.info(rec)
    else:
        st.success("✅ Great job! You're already practicing many eco-friendly habits!")
    
    # Additional context
    st.markdown("---")
    st.caption("💭 Note: This is an estimate based on machine learning predictions. Actual carbon footprint may vary based on many factors.")
