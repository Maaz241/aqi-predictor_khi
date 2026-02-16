from fastapi import FastAPI, HTTPException
import os
import sys
import pandas as pd
import pickle
from datetime import datetime

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data_ingestion.openweather_client import OpenWeatherClient
from src.database.mongodb_client import MongoDBClient
from src.feature_engineering.feature_processor import FeatureProcessor

app = FastAPI(title="Karachi AQI Predictor API")

# Load model assets
MODEL_PATH = "best_model.pkl"
ENCODER_PATH = "label_encoder.pkl"

def load_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(ENCODER_PATH, "rb") as f:
            encoder = pickle.load(f)
        return model, encoder
    return None, None

model, encoder = load_model()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Karachi AQI Predictor API", "status": "running"}

@app.get("/current")
def get_current_aqi():
    try:
        ow_client = OpenWeatherClient()
        data = ow_client.get_combined_data()
        category = FeatureProcessor.get_aqi_category(data['aqi'])
        data['aqi_category'] = category
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/forecast")
def get_forecast():
    if not model or not encoder:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")
    
    try:
        ow_client = OpenWeatherClient()
        current_data = ow_client.get_combined_data()
        forecast_features = ow_client.get_forecast_features()
        
        latest_pollutants = {k: current_data[k] for k in ['pm25', 'pm10', 'no2', 'o3', 'so2', 'co']}
        
        predictions = []
        features_order = ['pm25', 'pm10', 'no2', 'o3', 'so2', 'co', 
                         'temperature', 'humidity', 'pressure', 'wind_speed', 
                         'wind_deg', 'clouds', 'hour', 'day_of_week', 'month']
        
        for feat in forecast_features:
            input_data = {**feat, **latest_pollutants}
            dt = datetime.fromtimestamp(feat['timestamp'])
            input_data['hour'] = dt.hour
            input_data['day_of_week'] = dt.weekday()
            input_data['month'] = dt.month
            
            X_input = pd.DataFrame([input_data])[features_order]
            pred_encoded = model.predict(X_input)[0]
            pred_label = encoder.inverse_transform([pred_encoded])[0]
            
            predictions.append({
                "timestamp": feat['timestamp'],
                "time": dt.strftime("%Y-%m-%d %H:%M"),
                "temperature": feat['temperature'],
                "predicted_aqi_category": pred_label
            })
            
        return predictions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
def get_history(limit: int = 10):
    try:
        db_client = MongoDBClient()
        # Fetch latest records from MongoDB
        data = list(db_client.collection.find({}, {"_id": 0}).sort("timestamp", -1).limit(limit))
        db_client.close()
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
