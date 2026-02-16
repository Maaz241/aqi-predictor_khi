import sys
import os
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data_ingestion.openweather_client import OpenWeatherClient
from src.database.mongodb_client import MongoDBClient
from src.feature_engineering.feature_processor import FeatureProcessor

def run_pipeline():
    print(f"[{datetime.now()}] Starting Feature Pipeline...")
    
    # 1. Fetch Data
    ow_client = OpenWeatherClient()
    try:
        raw_data = ow_client.get_combined_data()
        print("Successfully fetched data from OpenWeather.")
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    # 2. Process Features
    # Convert single record to DataFrame for processor
    df = pd.DataFrame([raw_data])
    processor = FeatureProcessor()
    df_processed = processor.process_features(df)
    
    # 3. Store in MongoDB
    db_client = MongoDBClient()
    record = df_processed.to_dict('records')[0]
    
    # Upsert based on timestamp to avoid duplicates if run multiple times
    result = db_client.collection.update_one(
        {"timestamp": record["timestamp"]},
        {"$set": record},
        upsert=True
    )
    
    if result.upserted_id:
        print(f"Successfully stored NEW processed feature in MongoDB. ID: {result.upserted_id}")
    elif result.modified_count > 0:
        print("Updated existing record in MongoDB.")
    else:
        print("Record already exists and is up-to-date. No changes made.")
        
    db_client.close()

if __name__ == "__main__":
    run_pipeline()
