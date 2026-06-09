import json
import sys
from pathlib import Path

import joblib
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from carbon.features import build_feature_vector  # noqa: E402
from carbon.insights import get_ai_insights  # noqa: E402
from carbon.paths import COLUMNS_FILE, META_FILE, MODEL_FILE, SCALER_FILE  # noqa: E402


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    feature_cols = joblib.load(COLUMNS_FILE)
    meta = json.loads(META_FILE.read_text(encoding="utf-8")) if META_FILE.exists() else {}
    return model, scaler, feature_cols, meta


def render_ai_insights(insights: dict) -> None:
    st.markdown("**Main drivers**")
    for item in insights.get("main_drivers", []):
        st.markdown(f"- {item}")

    st.markdown("**Actions**")
    for item in insights.get("actions", []):
        st.markdown(f"- {item}")

    st.markdown("**Reality check**")
    st.info(insights.get("reality_check", ""))


model, scaler, feature_cols, meta = load_artifacts()

st.title("Carbon Footprint Calculator")
st.markdown(
    "Calculate your personal carbon emissions based on your lifestyle and habits."
)

with st.sidebar:
    st.subheader("Model")
    st.caption("XGBoost tuned with Optuna")
    if meta:
        st.metric("Test R²", f"{meta.get('test_r2', 0):.4f}")
        st.caption(f"Best trial #{meta.get('best_trial', '—')}")

tab1, tab2, tab3 = st.tabs(["Personal Info", "Lifestyle", "Transportation & Energy"])

with tab1:
    sex = st.selectbox("Gender", ["Male", "Female"])
    body_type = st.selectbox(
        "Body Type", ["Normal", "Obese", "Overweight", "Underweight"]
    )
    monthly_grocery = st.number_input("Monthly Grocery Bill ($)", 0, 1000, 200)

with tab2:
    diet = st.selectbox("Diet Type", ["Omnivore", "Pescatarian", "Vegan", "Vegetarian"])
    shower_frequency = st.selectbox(
        "Shower Frequency",
        ["Daily", "Less Frequently", "More Frequently", "Twice a Day"],
    )
    social_activity = st.selectbox("Social Activity", ["Never", "Often", "Sometimes"])
    tv_pc_hours = st.slider("TV/PC Daily Hours", 0, 24, 4)
    internet_hours = st.slider("Internet Daily Hours", 0, 24, 3)
    new_clothes = st.number_input("New Clothes Monthly", 0, 50, 2)
    waste_bags = st.number_input("Waste Bags Weekly", 0, 20, 3)
    waste_bag_size = st.selectbox(
        "Waste Bag Size", ["Small", "Medium", "Large", "Extra Large"]
    )
    recycling = st.multiselect("Recycling", ["Glass", "Metal", "Paper", "Plastic"])
    cooking_methods = st.multiselect(
        "Cooking Methods", ["Airfryer", "Grill", "Microwave", "Oven", "Stove"]
    )

with tab3:
    transport = st.selectbox("Primary Transport", ["Private", "Public", "Walk/Bicycle"])
    vehicle_type = st.selectbox(
        "Vehicle Type", ["None", "Electric", "Hybrid", "LPG", "Petrol", "Unknown"]
    )
    vehicle_distance = st.number_input("Vehicle Monthly Distance (Km)", 0, 5000, 500)
    air_travel = st.selectbox(
        "Air Travel Frequency", ["Never", "Rarely", "Frequently", "Very Frequently"]
    )
    heating_source = st.selectbox(
        "Heating Source", ["Coal", "Electricity", "Natural Gas", "Wood"]
    )
    energy_efficiency = st.selectbox("Energy Efficiency", ["No", "Sometimes", "Yes"])

if st.button("Calculate Carbon Footprint", type="primary", use_container_width=True):
    input_df = build_feature_vector(
        feature_cols,
        scaler,
        sex=sex,
        monthly_grocery=monthly_grocery,
        vehicle_distance=vehicle_distance,
        waste_bags=waste_bags,
        tv_pc_hours=tv_pc_hours,
        new_clothes=new_clothes,
        internet_hours=internet_hours,
        body_type=body_type,
        diet=diet,
        shower_frequency=shower_frequency,
        heating_source=heating_source,
        transport=transport,
        social_activity=social_activity,
        air_travel=air_travel,
        waste_bag_size=waste_bag_size,
        vehicle_type=vehicle_type,
        recycling=recycling,
        cooking_methods=cooking_methods,
    )

    prediction = float(model.predict(input_df)[0])

    st.success(f"### Annual Carbon Footprint: {prediction:,.2f} kg CO₂")

    st.metric("Daily Avg", f"{prediction / 365:.2f} kg")
    st.metric("Monthly Avg", f"{prediction / 12:.2f} kg")

    st.subheader("AI Expert Insights (Powered by Ollama)")

    user_data = {
        "predicted_annual_kg_co2": round(prediction, 2),
        "predicted_daily_kg_co2": round(prediction / 365, 2),
        "predicted_monthly_kg_co2": round(prediction / 12, 2),
        "gender": sex,
        "body_type": body_type,
        "monthly_grocery_bill_usd": monthly_grocery,
        "diet": diet,
        "shower_frequency": shower_frequency,
        "social_activity": social_activity,
        "tv_pc_hours_daily": tv_pc_hours,
        "internet_hours_daily": internet_hours,
        "new_clothes_monthly": new_clothes,
        "waste_bags_weekly": waste_bags,
        "waste_bag_size": waste_bag_size,
        "recycling": recycling,
        "cooking_methods": cooking_methods,
        "primary_transport": transport,
        "vehicle_type": vehicle_type,
        "vehicle_monthly_distance_km": vehicle_distance,
        "air_travel_frequency": air_travel,
        "heating_source": heating_source,
        "energy_efficiency": energy_efficiency,
    }

    with st.spinner("AI is analyzing your lifestyle..."):
        try:
            insights = get_ai_insights(user_data)
            render_ai_insights(insights)
        except (json.JSONDecodeError, KeyError, Exception) as exc:
            st.warning("Could not parse AI insights. Try again.")
            st.caption(str(exc))

    st.caption(
        "This is an estimate from an XGBoost model (Optuna-tuned). "
        "Real emissions depend on many real-world factors."
    )
