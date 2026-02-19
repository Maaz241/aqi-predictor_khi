import pandas as pd


class FeatureProcessor:
    @staticmethod
    def get_aqi_category(aqi_value):
        """Map AQI values from mixed sources into one OpenWeather-style 5-class set."""
        try:
            value = float(aqi_value)
        except (TypeError, ValueError):
            return None

        if pd.isna(value):
            return None

        # OpenWeather current-air API returns discrete AQI index in [1, 5].
        if 1 <= value <= 5 and value.is_integer():
            openweather_labels = {
                1: "Good",
                2: "Fair",
                3: "Moderate",
                4: "Poor",
                5: "Very Poor",
            }
            return openweather_labels[int(value)]

        # Historical numeric AQI scale collapsed into the same 5 OpenWeather-style buckets.
        if value <= 50:
            return "Good"
        if value <= 100:
            return "Fair"
        if value <= 150:
            return "Moderate"
        if value <= 200:
            return "Poor"
        return "Very Poor"

    @staticmethod
    def process_features(df):
        """Compute time-based and derived features."""
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            try:
                vals = pd.to_numeric(df["timestamp"], errors="coerce")
                if not vals.isna().any():
                    unit = "ms" if vals.iloc[0] > 1e11 else "s"
                    df["timestamp"] = pd.to_datetime(vals, unit=unit)
                else:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
            except Exception:
                df["timestamp"] = pd.to_datetime(df["timestamp"])

        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["month"] = df["timestamp"].dt.month

        df["aqi_category"] = df["aqi"].apply(FeatureProcessor.get_aqi_category)

        df = df.sort_values("timestamp")
        df["aqi_change_rate"] = df["aqi"].diff().fillna(0)

        return df
