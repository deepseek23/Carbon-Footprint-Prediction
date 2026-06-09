"""Feature engineering for inference."""

import pandas as pd

NUMERIC_COLS = [
    "Monthly Grocery Bill",
    "Vehicle Monthly Distance Km",
    "Waste Bag Weekly Count",
    "How Long TV PC Daily Hour",
    "How Many New Clothes Monthly",
    "How Long Internet Daily Hour",
]

CATEGORICAL_MAPS = {
    "body_type": {
        "Obese": "Body Type_obese",
        "Overweight": "Body Type_overweight",
        "Underweight": "Body Type_underweight",
    },
    "diet": {"Vegetarian": "Diet_vegetarian"},
    "shower_frequency": {"Twice a Day": "How Often Shower_twice a day"},
    "heating_source": {"Electricity": "Heating Energy Source_electricity"},
    "transport": {
        "Public": "Transport_public",
        "Walk/Bicycle": "Transport_walk/bicycle",
    },
    "social_activity": {"Often": "Social Activity_often"},
    "air_travel": {
        "Never": "Frequency of Traveling by Air_never",
        "Rarely": "Frequency of Traveling by Air_rarely",
        "Very Frequently": "Frequency of Traveling by Air_very frequently",
    },
    "waste_bag_size": {
        "Large": "Waste Bag Size_large",
        "Medium": "Waste Bag Size_medium",
        "Small": "Waste Bag Size_small",
    },
    "vehicle_type": {
        "Electric": "Vehicle Type_electric",
        "Hybrid": "Vehicle Type_hybrid",
        "LPG": "Vehicle Type_lpg",
        "Petrol": "Vehicle Type_petrol",
        "Unknown": "Vehicle Type_unknown",
    },
    "recycling": {
        "Glass": "Glass",
        "Metal": "Metal",
        "Paper": "Paper",
        "Plastic": "Plastic",
    },
    "cooking_methods": {"Oven": "Oven"},
}


def build_feature_vector(
    feature_cols,
    scaler,
    *,
    sex,
    monthly_grocery,
    vehicle_distance,
    waste_bags,
    tv_pc_hours,
    new_clothes,
    internet_hours,
    body_type,
    diet,
    shower_frequency,
    heating_source,
    transport,
    social_activity,
    air_travel,
    waste_bag_size,
    vehicle_type,
    recycling,
    cooking_methods,
):
    row = {col: 0.0 for col in feature_cols}
    row["Sex"] = 1.0 if sex == "Female" else 0.0

    raw_numeric = {
        "Monthly Grocery Bill": monthly_grocery,
        "Vehicle Monthly Distance Km": vehicle_distance,
        "Waste Bag Weekly Count": waste_bags,
        "How Long TV PC Daily Hour": tv_pc_hours,
        "How Many New Clothes Monthly": new_clothes,
        "How Long Internet Daily Hour": internet_hours,
    }
    scaled = scaler.transform(pd.DataFrame([raw_numeric])[NUMERIC_COLS])[0]
    for col, value in zip(NUMERIC_COLS, scaled):
        row[col] = float(value)

    selections = {
        "body_type": body_type,
        "diet": diet,
        "shower_frequency": shower_frequency,
        "heating_source": heating_source,
        "transport": transport,
        "social_activity": social_activity,
        "air_travel": air_travel,
        "waste_bag_size": waste_bag_size,
        "vehicle_type": vehicle_type,
    }
    for group, value in selections.items():
        col_name = CATEGORICAL_MAPS[group].get(value)
        if col_name and col_name in row:
            row[col_name] = 1.0

    for item in recycling:
        col_name = CATEGORICAL_MAPS["recycling"].get(item)
        if col_name and col_name in row:
            row[col_name] = 1.0

    for item in cooking_methods:
        col_name = CATEGORICAL_MAPS["cooking_methods"].get(item)
        if col_name and col_name in row:
            row[col_name] = 1.0

    return pd.DataFrame([row])[feature_cols]
