import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys
import shap
import matplotlib.pyplot as plt
from datetime import datetime

# ======================================================
# PATH SETUP
# ======================================================

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data_ingestion.openweather_client import OpenWeatherClient
from src.feature_engineering.feature_processor import FeatureProcessor

# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title="Karachi AQI Predictor",
    layout="wide"
)

# ======================================================
# LOAD MODEL + ENCODER
# ======================================================

@st.cache_resource
def load_model_assets():
    model, encoder = None, None

    if os.path.exists("best_model.pkl"):
        with open("best_model.pkl", "rb") as f:
            model = pickle.load(f)

    if os.path.exists("label_encoder.pkl"):
        with open("label_encoder.pkl", "rb") as f:
            encoder = pickle.load(f)

    return model, encoder


def load_metrics():
    if os.path.exists("metrics.json"):
        import json
        with open("metrics.json", "r") as f:
            return json.load(f)
    return None


model, encoder = load_model_assets()
metrics = load_metrics()

# ======================================================
# SIDEBAR METRICS
# ======================================================

if metrics:
    st.sidebar.header("📊 Model Comparison")

    if isinstance(metrics, dict):
        metrics = [metrics]

    df = pd.DataFrame(metrics)
    st.sidebar.dataframe(df.set_index("model"), use_container_width=True)

    best = df.loc[df["f1_score"].idxmax()]
    st.sidebar.success(f"Best Model: {best['model']}")

# ======================================================
# AQI HELPERS
# ======================================================

def get_aqi_color(category):
    return {
        "Good": "#00e400",
        "Fair": "#ffff00",
        "Moderate": "#ff7e00",
        "Poor": "#ff0000",
        "Very Poor": "#8f3f97"
    }.get(category, "#ffffff")


def get_openweather_aqi_label(idx):
    return {
        1: "Good",
        2: "Fair",
        3: "Moderate",
        4: "Poor",
        5: "Very Poor"
    }.get(idx, "Unknown")


# ======================================================
# MAIN UI
# ======================================================

st.title("🌍 Karachi Air Quality Predictor")

if st.sidebar.button("Fetch Real-time Data"):
    with st.spinner("Fetching data..."):
        client = OpenWeatherClient()
        st.session_state["current_data"] = client.get_combined_data()
        st.success("Data updated")

# ======================================================
# DISPLAY DATA
# ======================================================

if "current_data" not in st.session_state:
    st.info("Click **Fetch Real-time Data** to start.")
    st.stop()

data = st.session_state["current_data"]

col1, col2, col3 = st.columns(3)

# ------------------ AQI ------------------
with col1:
    category = get_openweather_aqi_label(data["aqi"])
    st.metric("Current AQI (OpenWeather)", f"{data['aqi']} / 5")
    st.markdown(
        f"### Status: <span style='color:{get_aqi_color(category)}'>{category}</span>",
        unsafe_allow_html=True
    )

    if category in ["Poor", "Very Poor"]:
        st.warning("🚨 Air quality is unhealthy")

# ------------------ POLLUTION ------------------
with col2:
    st.subheader("Pollutants")
    st.table(pd.DataFrame({
        "Pollutant": ["PM2.5", "PM10", "NO2", "O3", "SO2", "CO"],
        "Value": [
            data["pm25"], data["pm10"], data["no2"],
            data["o3"], data["so2"], data["co"]
        ]
    }))

# ------------------ WEATHER ------------------
with col3:
    st.subheader("Weather")
    st.write(f"🌡️ Temperature: {data['temperature']} °C")
    st.write(f"💧 Humidity: {data['humidity']} %")
    st.write(f"🌬️ Wind: {data['wind_speed']} m/s")

# ======================================================
# FORECAST
# ======================================================

st.divider()
st.header("🔮 3-Day AQI Forecast")

features_order = [
    "pm25", "pm10", "no2", "o3", "so2", "co",
    "temperature", "humidity", "pressure",
    "wind_speed", "wind_deg", "clouds",
    "hour", "day_of_week", "month"
]

if model and encoder:
    client = OpenWeatherClient()
    forecast = client.get_forecast_features()

    latest_pollutants = {k: data[k] for k in ["pm25","pm10","no2","o3","so2","co"]}
    rows = []

    for f in forecast:
        merged = {**f, **latest_pollutants}
        dt = datetime.fromtimestamp(f["timestamp"])

        merged["hour"] = dt.hour
        merged["day_of_week"] = dt.weekday()
        merged["month"] = dt.month

        X = pd.DataFrame([merged])[features_order]
        pred = model.predict(X)[0]
        label = encoder.inverse_transform([pred])[0]

        rows.append({
            "Time": dt.strftime("%Y-%m-%d %H:%M"),
            "Predicted AQI": label
        })

    st.table(pd.DataFrame(rows))

# ======================================================
# SHAP EXPLANATION (FIXED)
# ======================================================

st.divider()
st.header("🔍 Feature Importance (SHAP)")

current = dict(data)
dt = datetime.fromtimestamp(data["timestamp"])

current["hour"] = dt.hour
current["day_of_week"] = dt.weekday()
current["month"] = dt.month

X_current = pd.DataFrame([current])[features_order]
X_current = X_current.apply(pd.to_numeric)

try:
    explainer = shap.Explainer(model)
    shap_values = explainer(X_current)

    fig = plt.figure()
    shap.plots.bar(shap_values, show=False)
    st.pyplot(fig)

except Exception as e:
    st.error(f"SHAP Error: {e}")
