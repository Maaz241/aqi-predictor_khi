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
# LOAD MODEL + ENCODER + POLLUTANT MODELS
# ===========================

@st.cache_resource
def load_model_assets():
    model_path = "best_model.pkl"
    encoder_path = "label_encoder.pkl"
    pollutant_models_path = "pollutant_models.pkl"
    
    model, encoder, pollutant_models = None, None, None
    
    if os.path.exists(model_path) and os.path.exists(encoder_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(encoder_path, "rb") as f:
            encoder = pickle.load(f)
    
    if os.path.exists(pollutant_models_path):
        with open(pollutant_models_path, "rb") as f:
            pollutant_models = pickle.load(f)
            
    return model, encoder, pollutant_models


def load_metrics():
    metrics_path = "metrics.json"
    if os.path.exists(metrics_path):
        import json
        with open(metrics_path, "r") as f:
            return json.load(f)
    return None


model, encoder, pollutant_models = load_model_assets()
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
    display_cols = ['model', 'accuracy', 'f1_score']
    display_cols = [col for col in display_cols if col in metrics_df.columns]
    st.sidebar.dataframe(metrics_df[display_cols].set_index('model'), use_container_width=True)

    best_primary_model = metrics_df.loc[metrics_df['f1_score'].idxmax()]
    horizon = int(best_primary_model.get('forecast_horizon_steps', 1))
    st.sidebar.success(
        f"Best Forecast Model (+{horizon} step): {best_primary_model['model']} "
        f"(F1: {best_primary_model['f1_score']:.3f})"
    )
    
    st.sidebar.divider()

   

# ===========================
# COLOR MAPPING (5 CLASSES)
# ===========================

def normalize_aqi_category(category):
    mapping = {
        "Good": "Good",
        "Fair": "Fair",
        "Moderate": "Moderate",
        "Poor": "Poor",
        "Very Poor": "Very Poor",
        # Backward compatibility for older model artifacts
        "Unhealthy for Sensitive Groups": "Moderate",
        "Unhealthy": "Poor",
        "Very Unhealthy": "Very Poor",
        "Hazardous": "Very Poor",
    }
    return mapping.get(category, category)


def get_aqi_color(category):
    category = normalize_aqi_category(category)
    colors = {
        "Good": "#00e400",
        "Fair": "#ffff00",
        "Moderate": "#ff7e00",
        "Poor": "#ff0000",
        "Very Poor": "#8f3f97",
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


def _extract_shap_vector(raw_values, class_index, feature_count):
    """Normalize SHAP output shape across model and explainer types."""
    if isinstance(raw_values, list):
        if not raw_values:
            return None
        class_idx = min(class_index, len(raw_values) - 1)
        arr = np.array(raw_values[class_idx])
        if arr.ndim == 2:
            return arr[0]
        if arr.ndim == 1:
            return arr
        return None

    arr = np.array(raw_values)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        return arr[0]
    if arr.ndim == 3:
        # Typical shape: (n_samples, n_features, n_classes)
        if arr.shape[0] == 1 and arr.shape[1] == feature_count:
            class_idx = min(class_index, arr.shape[2] - 1)
            return arr[0, :, class_idx]
        # Alternate shape: (n_samples, n_classes, n_features)
        if arr.shape[0] == 1 and arr.shape[2] == feature_count:
            class_idx = min(class_index, arr.shape[1] - 1)
            return arr[0, class_idx, :]
        # Alternate shape: (n_classes, n_samples, n_features)
        if arr.shape[2] == feature_count:
            class_idx = min(class_index, arr.shape[0] - 1)
            return arr[class_idx, 0, :]
    return None


def compute_shap_contributions(model_obj, X_input, class_index, feature_names):
    """Compute per-feature SHAP contributions for one prediction row."""
    try:
        explainer = shap.Explainer(model_obj)
        explanation = explainer(X_input)
        raw_values = explanation.values
    except Exception:
        try:
            explainer = shap.TreeExplainer(model_obj)
            raw_values = explainer.shap_values(X_input)
        except Exception:
            return None

    shap_vector = _extract_shap_vector(raw_values, class_index, len(feature_names))
    if shap_vector is None or len(shap_vector) != len(feature_names):
        return None

    contributions = pd.DataFrame({
        "feature": feature_names,
        "shap_value": shap_vector
    })
    contributions["abs_shap"] = contributions["shap_value"].abs()
    contributions = contributions.sort_values("abs_shap", ascending=False).reset_index(drop=True)
    return contributions


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
    current_model_input = None
    current_pred_encoded = None
    model_category = None
    
    col1, col2, col3 = st.columns(3)
    
    # ---------------------------
    # CURRENT AQI
    # ---------------------------
    with col1:
        openweather_index = int(round(float(data['aqi'])))
        openweather_category = normalize_aqi_category(get_openweather_aqi_label(openweather_index))
        st.metric("Current AQI Index (OpenWeather)", f"{openweather_index} / 5")

        # Keep current status tied to OpenWeather index so number and label always match.
        category = openweather_category

        # Show model nowcast separately for comparison.
        if model and encoder:
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
            model_category = normalize_aqi_category(encoder.inverse_transform([pred_encoded])[0])
            current_model_input = X_current
            current_pred_encoded = int(pred_encoded)
            st.caption(f"Model nowcast: {model_category}")
        
        st.markdown(
            f"### Status: <span style='color:{get_aqi_color(category)}'>{category}</span>",
            unsafe_allow_html=True
        )
        
        if category in ["Poor", "Very Poor"]:
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
    # SHAP ANALYSIS (CURRENT NOWCAST)
    # ===========================
    if model and encoder and current_model_input is not None and current_pred_encoded is not None:
        st.divider()
        st.subheader("SHAP Feature Analysis (Model Nowcast)")

        shap_df = compute_shap_contributions(
            model_obj=model,
            X_input=current_model_input,
            class_index=current_pred_encoded,
            feature_names=current_model_input.columns.tolist()
        )

        if shap_df is None or shap_df.empty:
            st.info("SHAP analysis is not available for the current model artifact.")
        else:
            top_n = shap_df.head(10).iloc[::-1]
            fig, ax = plt.subplots(figsize=(8, 5))
            colors = ["#d62728" if v > 0 else "#1f77b4" for v in top_n["shap_value"]]
            ax.barh(top_n["feature"], top_n["shap_value"], color=colors)
            ax.axvline(0, color="black", linewidth=1)
            ax.set_xlabel("SHAP value")
            ax.set_ylabel("Feature")
            if model_category:
                ax.set_title(f"Top impacts toward model class: {model_category}")
            else:
                ax.set_title("Top SHAP feature impacts")
            st.pyplot(fig, clear_figure=True)
            st.caption("Positive values push prediction toward the shown class; negative values push away.")
            st.dataframe(
                shap_df[["feature", "shap_value"]].head(10),
                use_container_width=True
            )

    # ===========================
    # 3-DAY FORECAST
    # ===========================

    st.divider()
    st.header("🔮 3-Day Prediction Forecast")

    if model and encoder and pollutant_models:
        from src.forecasting.autoregressive_forecaster import AutoregressiveForecaster
        
        # Get weather forecast
        ow_client = OpenWeatherClient()
        forecast_features = ow_client.get_forecast_features()
        
        # Create forecaster
        forecaster = AutoregressiveForecaster(model, pollutant_models, encoder)
        
        # Get autoregressive predictions
        predictions = forecaster.predict_3day_forecast(forecast_features)
        
        # Display predictions in table
        forecast_df = pd.DataFrame([
            {
                "Time": pred['timestamp'],
                "Temp": f"{pred['temperature']:.1f}°C",
                "PM2.5": f"{pred['pm25']:.1f}",
                "PM10": f"{pred['pm10']:.1f}",
                "NO2": f"{pred['no2']:.2f}",
                "O3": f"{pred['o3']:.1f}",
                "AQI Status": normalize_aqi_category(pred['aqi_category'])
            }
            for pred in predictions
        ])
        
        st.dataframe(forecast_df, use_container_width=True, height=400)
        
        # Show note about methodology
        with st.expander("ℹ️ How are these predictions made?"):
            st.write("""
            **Autoregressive Forecasting Methodology:**
            
            1. **Historical Trends**: Uses the last 3 time steps of pollutant data as lag features
            2. **Weather Forecast**: Incorporates temperature, humidity, wind, etc. from OpenWeather API
            3. **Iterative Prediction**: 
               - Predicts pollutants for Day 1 using historical data + weather
               - Uses Day 1 predictions as lag features for Day 2
               - Uses Day 2 predictions as lag features for Day 3
            4. **AQI Classification**: Predicts AQI category using predicted pollutants + weather
            
            """)
    
    elif model and encoder:
        st.warning("⚠️ Pollutant prediction models not found. Showing simplified forecast (assumes constant pollutant levels).")
        
        # Fallback to old method
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
            pred_label = normalize_aqi_category(encoder.inverse_transform([pred_encoded])[0])

            predictions.append({
                "Time": dt.strftime("%Y-%m-%d %H:%M"),
                "Weather": f"{feat['temperature']}°C",
                "Predicted AQI Status": pred_label
            })

        st.table(pd.DataFrame(predictions))
    
    else:
        st.info("Model not found. Please run the training pipeline first.")

else:
    st.info("Click 'Fetch Real-time Data' to start.")
