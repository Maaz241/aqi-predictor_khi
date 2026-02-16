import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.database.mongodb_client import MongoDBClient
from src.feature_engineering.feature_processor import FeatureProcessor

def backfill():
    print("Starting data backfill...")
    file_path = "Historical_data_karachi.xlsx"
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    # Load data
    df = pd.read_excel(file_path)
    print(f"Loaded {len(df)} records from {file_path}")

    # Process features
    processor = FeatureProcessor()
    df_processed = processor.process_features(df)
    
    # Convert to list of dicts for MongoDB
    records = df_processed.to_dict('records')
    
    # Store in MongoDB
    db_client = MongoDBClient()
    db_client.collection.delete_many({}) # Clear existing for fresh start
    db_client.insert_many(records)
    print(f"Successfully backfilled {len(records)} records into MongoDB.")
    db_client.close()

if __name__ == "__main__":
    backfill()
