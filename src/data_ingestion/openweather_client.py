import os
import requests
from dotenv import load_dotenv

load_dotenv()

class OpenWeatherClient:
    def __init__(self):
        self.api_key = os.getenv("OPENWEATHER_API_KEY")
        self.lat = os.getenv("LAT")
        self.lon = os.getenv("LON")
        self.pollution_url = f"https://api.openweathermap.org/data/2.5/air_pollution"
        self.weather_url = f"https://api.openweathermap.org/data/2.5/weather"

    def fetch_pollution_data(self):
        """Fetches air pollution data from OpenWeather API."""
        params = {
            "lat": self.lat,
            "lon": self.lon,
            "appid": self.api_key
        }
        response = requests.get(self.pollution_url, params=params)
        response.raise_for_status()
        return response.json()

    def fetch_weather_data(self):
        """Fetches current weather data from OpenWeather API."""
        params = {
            "lat": self.lat,
            "lon": self.lon,
            "units": "metric",
            "appid": self.api_key
        }
        response = requests.get(self.weather_url, params=params)
        response.raise_for_status()
        return response.json()

    def get_combined_data(self):
        """Combines pollution and weather data into a single record."""
        pollution = self.fetch_pollution_data()
        weather = self.fetch_weather_data()
        
        # Extract components
        p_list = pollution['list'][0]
        w_main = weather['main']
        w_wind = weather['wind']
        w_clouds = weather['clouds']
        
        combined = {
            "timestamp": p_list['dt'],
            "pm25": p_list['components']['pm2_5'],
            "pm10": p_list['components']['pm10'],
            "no2": p_list['components']['no2'],
            "o3": p_list['components']['o3'],
            "so2": p_list['components']['so2'],
            "co": p_list['components']['co'],
            "temperature": w_main['temp'],
            "humidity": w_main['humidity'],
            "pressure": w_main['pressure'],
            "wind_speed": w_wind['speed'],
            "wind_deg": w_wind['deg'],
            "clouds": w_clouds['all'],
            "aqi": p_list['main']['aqi'] # OpenWeather AQI scale (1-5)
        }
        return combined

    def fetch_forecast_data(self):
        """Fetches 5-day / 3-hour forecast data."""
        url = "https://api.openweathermap.org/data/2.5/forecast"
        params = {
            "lat": self.lat,
            "lon": self.lon,
            "units": "metric",
            "appid": self.api_key
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def get_forecast_features(self):
        """Prepares features for the next 3 days from the forecast."""
        forecast = self.fetch_forecast_data()
        forecast_list = forecast['list']
        
        features_list = []
        for item in forecast_list[:24]: # 24 * 3 hours = 72 hours = 3 days
            features = {
                "timestamp": item['dt'],
                "temperature": item['main']['temp'],
                "humidity": item['main']['humidity'],
                "pressure": item['main']['pressure'],
                "wind_speed": item['wind']['speed'],
                "wind_deg": item['wind']['deg'],
                "clouds": item['clouds']['all']
            }
            # For forecast, we don't have pollutants, so we might need to 
            # approximate or use the model to predict step by step.
            # However, the task asks for predictions for next 3 days.
            # Usually, this means using weather forecast as features.
            # But the model was trained with pollutants.
            # Suggestion: For future predictions, we use the latest pollutant values 
            # and the forecast weather values.
            features_list.append(features)
        return features_list
