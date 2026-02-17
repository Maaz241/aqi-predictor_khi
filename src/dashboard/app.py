import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys
import shap
import matplotlib.pyplot as plt
from datetime import datetime

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data_ingestion.openweather_client import OpenWeatherClient
from src.feature_engineering.feature_processor import FeatureProcessor

# Page config
st.set_page_config(page_title="Karachi AQI Predictor", layout="wide")

# ===========================
# LOAD MODEL + ENCODER
# ===========================

@st.cache_resource
def load_model_assets():
    model_path = "best_model.pkl"
    encoder_path = "label_encoder.pkl"
    
    model, encoder = None, None
    
    if os.path.exists(model_path) and os.path.exists(encoder_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(encoder_path, "rb") as f:
            encoder = pickle.load(f)
            
    return model, encoder


def load_metrics():
    metrics_path = "metrics.json"
    if os.path.exists(metrics_path):
        import json
        with open(metrics_path, "r") as f:
            return json.load(f)
    return None


model, encoder = load_model_assets()
metrics = load_metrics()

# ===========================
# SIDEBAR - MODEL METRICS
# ===========================

if metrics:
    st.sidebar.divider()
    st.sidebar.header("📊 Model Comparison")
    
    if isinstance(metrics, dict):
        metrics = [metrics]
    
    metrics_df = pd.DataFrame(metrics)
    st.sidebar.dataframe(metrics_df.set_index('model'), use_container_width=True)
    
    best_model = metrics_df.loc[metrics_df['f1_score'].idxmax()]
    st.sidebar.success(f"Best Model: {best_model['model']}")
    
    st.sidebar.divider()

   

# ===========================
# COLOR MAPPING (5 CLASSES)
# ===========================

def get_aqi_color(category):
    colors = {
        # OpenWeather classes
        "Good": "#00e400",
        "Fair": "#ffff00",
        "Moderate": "#ff7e00",
        "Poor": "#ff0000",
        "Very Poor": "#8f3f97",
        # EPA classes (used by ML model)
        "Unhealthy for Sensitive Groups": "#ff7e00",
        "Unhealthy": "#ff0000",
        "Very Unhealthy": "#8f3f97",
        "Hazardous": "#7e0023"
    }
    return colors.get(category, "#ffffff")


def get_openweather_aqi_label(aqi_index):
    mapping = {
        1: "Good",
        2: "Fair",
        3: "Moderate",
        4: "Poor",
        5: "Very Poor"
    }
    return mapping.get(aqi_index, "Unknown")


# ===========================
# MAIN TITLE
# ===========================

st.title("🌍 Karachi Air Quality Predictor")


# ===========================
# FETCH REAL-TIME DATA
# ===========================

if st.sidebar.button("Fetch Real-time Data"):
    with st.spinner("Fetching data..."):
        ow_client = OpenWeatherClient()
        current_data = ow_client.get_combined_data()
        st.session_state['current_data'] = current_data
        st.success("Data updated!")

# ===========================
# DISPLAY DATA
# ===========================

if 'current_data' in st.session_state:
    data = st.session_state['current_data']
    
    col1, col2, col3 = st.columns(3)
    
    # ---------------------------
    # CURRENT AQI
    # ---------------------------
    with col1:
        st.metric("Current AQI Index (OpenWeather)", f"{data['aqi']} / 5")
        
        # Use ML model to predict current AQI category for consistency with forecast
        if model and encoder:
            # Prepare current data for prediction
            current_input = {**data}
            current_dt = datetime.fromtimestamp(data['timestamp'])
            current_input['hour'] = current_dt.hour
            current_input['day_of_week'] = current_dt.weekday()
            current_input['month'] = current_dt.month
            
            features_order = [
                'pm25', 'pm10', 'no2', 'o3', 'so2', 'co',
                'temperature', 'humidity', 'pressure',
                'wind_speed', 'wind_deg', 'clouds',
                'hour', 'day_of_week', 'month'
            ]
            
            X_current = pd.DataFrame([current_input])[features_order]
            pred_encoded = model.predict(X_current)[0]
            category = encoder.inverse_transform([pred_encoded])[0]
        else:
            # Fallback to OpenWeather label if model not available
            category = get_openweather_aqi_label(data['aqi'])
        
        st.markdown(
            f"### Status: <span style='color:{get_aqi_color(category)}'>{category}</span>",
            unsafe_allow_html=True
        )
        
        if category in ["Poor", "Very Poor", "Unhealthy", "Very Unhealthy", "Hazardous", "Unhealthy for Sensitive Groups"]:
            st.warning("🚨 ALERT: Air quality is unhealthy. Limit outdoor exposure.")

    # ---------------------------
    # POLLUTANTS
    # ---------------------------
    with col2:
        st.write("### Pollutant Breakdown")
        pollution_df = pd.DataFrame({
            "Pollutant": ["PM2.5", "PM10", "NO2", "O3", "SO2", "CO"],
            "Value": [
                data['pm25'],
                data['pm10'],
                data['no2'],
                data['o3'],
                data['so2'],
                data['co']
            ]
        })
        st.table(pollution_df)

    # ---------------------------
    # WEATHER
    # ---------------------------
    with col3:
        st.write("### Weather Conditions")
        st.write(f"🌡️ Temperature: {data['temperature']}°C")
        st.write(f"💧 Humidity: {data['humidity']}%")
        st.write(f"🌬️ Wind: {data['wind_speed']} m/s ({data['wind_deg']}°)")

    # ===========================
    # 3-DAY FORECAST
    # ===========================

    st.divider()
    st.header("🔮 3-Day Prediction Forecast")

    if model and encoder:
        ow_client = OpenWeatherClient()
        forecast_features = ow_client.get_forecast_features()

        latest_pollutants = {
            k: data[k] for k in ['pm25', 'pm10', 'no2', 'o3', 'so2', 'co']
        }

        predictions = []

        for feat in forecast_features:

            input_data = {**feat, **latest_pollutants}

            dt = datetime.fromtimestamp(feat['timestamp'])
            input_data['hour'] = dt.hour
            input_data['day_of_week'] = dt.weekday()
            input_data['month'] = dt.month

            features_order = [
                'pm25', 'pm10', 'no2', 'o3', 'so2', 'co',
                'temperature', 'humidity', 'pressure',
                'wind_speed', 'wind_deg', 'clouds',
                'hour', 'day_of_week', 'month'
            ]

            X_input = pd.DataFrame([input_data])[features_order]

            pred_encoded = model.predict(X_input)[0]
            pred_label = encoder.inverse_transform([pred_encoded])[0]

            predictions.append({
                "Time": dt.strftime("%Y-%m-%d %H:%M"),
                "Weather": f"{feat['temperature']}°C",
                "Predicted AQI Status": pred_label
            })

        st.table(pd.DataFrame(predictions))
