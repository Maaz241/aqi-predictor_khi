import pandas as pd
import datetime

class FeatureProcessor:
    @staticmethod
    def get_aqi_category(aqi_value):
        """Categorizes AQI value into classification buckets."""
        if aqi_value <= 50:
            return "Good"
        elif aqi_value <= 100:
            return "Moderate"
        elif aqi_value <= 150:
            return "Unhealthy for Sensitive Groups"
        elif aqi_value <= 200:
            return "Unhealthy"
        elif aqi_value <= 300:
            return "Very Unhealthy"
        else:
            return "Hazardous"

    @staticmethod
    def process_features(df):
        """Computes time-based and derived features."""
        # Convert timestamp if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s' if df['timestamp'].iloc[0] > 1e10 else None)
        
        # Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        
        # AQI Category
        df['aqi_category'] = df['aqi'].apply(FeatureProcessor.get_aqi_category)
        
        # Derived features: AQI change rate (requires sorting)
        df = df.sort_values('timestamp')
        df['aqi_change_rate'] = df['aqi'].diff().fillna(0)
        
        return df
