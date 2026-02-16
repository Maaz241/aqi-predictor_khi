# Karachi AQI Predictor

An end-to-end Air Quality Index (AQI) prediction system for Karachi, Pakistan. This project fetches real-time pollution and weather data, processes it into features, trains classification models, and displays predictions on an interactive Streamlit dashboard.

## Features
- **Data Ingestion**: Real-time fetching from OpenWeather Air Pollution and Weather APIs.
- **Feature Store**: MongoDB used for storing processed historical and real-time features.
- **Training Pipeline**: Random Forest and XGBoost classification models tracked with MLflow and DagsHub.
- **Streamlit Dashboard**:
    - Real-time AQI and pollutant breakdown.
    - 3-day AQI category forecast.
    - Feature importance visualizations using SHAP.
    - Hazardous AQI level alerts.
- **Automation**: GitHub Actions workflows for 4-hour feature updates and daily model retraining.

## Setup Instructions

### 1. Environment Variables
Ensure your `.env` file is configured with the following:
- `OPENWEATHER_API_KEY`: Your OpenWeather API key.
- `MONGODB_URI`: Your MongoDB connection string.
- `DATABASE_NAME`: `aqi`
- `LAT`: `24.8607`
- `LON`: `67.0011`
- `DAGSHUB_REPO_OWNER`, `DAGSHUB_REPO_NAME`, `DAGSHUB_TOKEN`: For model registry.
- `MLFLOW_TRACKING_URI`: Your DagsHub MLflow URI.

### 2. Installation
```bash
pip install -r requirements.txt
```

### 3. Data Backfill
Run the backfill script to populate MongoDB with historical data:
```bash
python scripts/backfill_data.py
```

### 4. Training
Train the models and log to MLflow:
```bash
python src/models/train.py
```

### 5. Running the Backend API
```bash
python src/api/main.py
```
The API will be available at `http://localhost:8000`. You can access the auto-generated documentation at `http://localhost:8000/docs`.

### 7. Running All Tests
```bash
python scripts/run_all.py
```
This script will run the feature pipeline, training pipeline, and Streamlit dashboard.

## Project Structure
- `src/`: Core source code.
- `scripts/`: Utility scripts (backfill).
- `.github/workflows/`: CI/CD pipelines.
- `notebooks/`: EDA and experimentation.
