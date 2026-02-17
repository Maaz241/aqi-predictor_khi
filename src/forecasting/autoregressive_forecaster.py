import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
from datetime import datetime

from src.database.mongodb_client import MongoDBClient


class AutoregressiveForecaster:
    """
    Autoregressive forecaster for 3-day AQI predictions.
    
    Uses historical pollutant trends to predict future pollutants,
    then uses those predictions as lag features for subsequent time steps.
    """
    
    def __init__(self, aqi_model, pollutant_models, label_encoder):
        """
        Initialize forecaster with models.
        
        Args:
            aqi_model: Trained AQI classification model
            pollutant_models: Dict of trained pollutant prediction models
            label_encoder: Label encoder for AQI categories
        """
        self.aqi_model = aqi_model
        self.pollutant_models = pollutant_models
        self.label_encoder = label_encoder
        self.pollutants = ['pm25', 'pm10', 'no2', 'o3', 'so2', 'co']
    
    def get_historical_context(self, n_lags=3):
        """
        Fetch last n records from MongoDB for lag features.
        
        Args:
            n_lags: Number of lag periods to fetch
            
        Returns:
            DataFrame with historical data
        """
        client = MongoDBClient()
        data = list(client.collection.find().sort('timestamp', -1).limit(n_lags))
        client.close()
        
        # Reverse to get chronological order
        data = data[::-1]
        df = pd.DataFrame(data)
        
        return df
    
    def calculate_rolling_stats(self, history_df, pollutant):
        """Calculate rolling statistics from historical data"""
        if len(history_df) >= 6:
            rolling_mean = history_df[pollutant].tail(6).mean()
            rolling_std = history_df[pollutant].tail(6).std()
        else:
            rolling_mean = history_df[pollutant].mean()
            rolling_std = history_df[pollutant].std()
        
        return rolling_mean, rolling_std
    
    def predict_pollutants(self, weather_forecast, lag_data, history_df):
        """
        Predict pollutants for next time step.
        
        Args:
            weather_forecast: Dict with weather features
            lag_data: Dict with lag features for each pollutant
            history_df: Historical data for rolling stats
            
        Returns:
            Dict with predicted pollutant levels
        """
        predicted_pollutants = {}
        
        for pollutant in self.pollutants:
            model_info = self.pollutant_models[pollutant]
            model = model_info['model']
            feature_cols = model_info['features']
            
            # Build feature vector
            features = {}
            
            # Lag features for this pollutant
            for i in range(1, 4):
                lag_key = f'{pollutant}_lag{i}'
                features[lag_key] = lag_data[pollutant][i-1] if i-1 < len(lag_data[pollutant]) else 0
            
            # Rolling statistics
            rolling_mean, rolling_std = self.calculate_rolling_stats(history_df, pollutant)
            features[f'{pollutant}_rolling_mean'] = rolling_mean
            features[f'{pollutant}_rolling_std'] = rolling_std
            
            # Lag features from other pollutants (lag1 only)
            for other_p in self.pollutants:
                if other_p != pollutant:
                    lag_key = f'{other_p}_lag1'
                    features[lag_key] = lag_data[other_p][0] if len(lag_data[other_p]) > 0 else 0
            
            # Weather features
            features['temperature'] = weather_forecast['temperature']
            features['humidity'] = weather_forecast['humidity']
            features['pressure'] = weather_forecast['pressure']
            features['wind_speed'] = weather_forecast['wind_speed']
            features['wind_deg'] = weather_forecast['wind_deg']
            features['clouds'] = weather_forecast['clouds']
            
            # Temporal features
            dt = datetime.fromtimestamp(weather_forecast['timestamp'])
            features['hour'] = dt.hour
            features['day_of_week'] = dt.weekday()
            features['month'] = dt.month
            
            # Create DataFrame with correct column order
            X = pd.DataFrame([features])[feature_cols]
            
            # Predict
            prediction = model.predict(X)[0]
            
            # Ensure non-negative
            prediction = max(0, prediction)
            
            predicted_pollutants[pollutant] = prediction
        
        return predicted_pollutants
    
    def predict_aqi(self, pollutants, weather, timestamp):
        """
        Predict AQI category given pollutants and weather.
        
        Args:
            pollutants: Dict with pollutant levels
            weather: Dict with weather features
            timestamp: Unix timestamp
            
        Returns:
            AQI category string
        """
        # Prepare features
        dt = datetime.fromtimestamp(timestamp)
        
        features = {
            **pollutants,
            'temperature': weather['temperature'],
            'humidity': weather['humidity'],
            'pressure': weather['pressure'],
            'wind_speed': weather['wind_speed'],
            'wind_deg': weather['wind_deg'],
            'clouds': weather['clouds'],
            'hour': dt.hour,
            'day_of_week': dt.weekday(),
            'month': dt.month
        }
        
        feature_order = [
            'pm25', 'pm10', 'no2', 'o3', 'so2', 'co',
            'temperature', 'humidity', 'pressure',
            'wind_speed', 'wind_deg', 'clouds',
            'hour', 'day_of_week', 'month'
        ]
        
        X = pd.DataFrame([features])[feature_order]
        
        # Predict
        pred_encoded = self.aqi_model.predict(X)[0]
        aqi_category = self.label_encoder.inverse_transform([pred_encoded])[0]
        
        return aqi_category
    
    def predict_3day_forecast(self, weather_forecast_list):
        """
        Main method: Autoregressive 3-day prediction.
        
        Args:
            weather_forecast_list: List of weather forecast dicts (24 points)
            
        Returns:
            List of prediction dicts with timestamp, pollutants, and AQI
        """
        # Get historical context
        history_df = self.get_historical_context(n_lags=6)
        
        # Initialize lag data with historical values
        lag_data = {pollutant: [] for pollutant in self.pollutants}
        
        for pollutant in self.pollutants:
            # Get last 3 values as initial lags
            lag_data[pollutant] = history_df[pollutant].tail(3).tolist()
        
        predictions = []
        
        for weather_forecast in weather_forecast_list:
            # Predict pollutants for this time step
            predicted_pollutants = self.predict_pollutants(
                weather_forecast, 
                lag_data, 
                history_df
            )
            
            # Predict AQI using predicted pollutants
            aqi_category = self.predict_aqi(
                predicted_pollutants,
                weather_forecast,
                weather_forecast['timestamp']
            )
            
            # Store prediction
            dt = datetime.fromtimestamp(weather_forecast['timestamp'])
            
            prediction = {
                'timestamp': dt.strftime("%Y-%m-%d %H:%M"),
                'datetime': dt,
                **predicted_pollutants,
                'temperature': weather_forecast['temperature'],
                'humidity': weather_forecast['humidity'],
                'wind_speed': weather_forecast['wind_speed'],
                'aqi_category': aqi_category
            }
            
            predictions.append(prediction)
            
            # Update lag data for next iteration
            for pollutant in self.pollutants:
                # Add new prediction and keep only last 3
                lag_data[pollutant].append(predicted_pollutants[pollutant])
                lag_data[pollutant] = lag_data[pollutant][-3:]
            
            # Add predicted values to history for rolling stats
            new_row = {**predicted_pollutants, 'timestamp': weather_forecast['timestamp']}
            history_df = pd.concat([history_df, pd.DataFrame([new_row])], ignore_index=True)
        
        return predictions
