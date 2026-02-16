import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import dagshub
from dotenv import load_dotenv
import json
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.database.mongodb_client import MongoDBClient

load_dotenv()

# DagsHub / MLflow Configuration
dagshub_token = os.getenv("DAGSHUB_TOKEN")
repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
repo_name = os.getenv("DAGSHUB_REPO_NAME")

if dagshub_token and repo_owner and repo_name:
    print("Initializing DagsHub...")
    dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    
    # Explicitly set credentials for MLflow
    os.environ['MLFLOW_TRACKING_USERNAME'] = repo_owner
    os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token

def train_models():
    print("Starting Training Pipeline...")
    
    # 1. Load Data from MongoDB
    db_client = MongoDBClient()
    data = db_client.fetch_all()
    db_client.close()
    
    if not data:
        print("No data found in MongoDB.")
        return
    
    df = pd.DataFrame(data)
    
    # 2. Preprocessing
    # Features: pollutants, weather, time-based features
    features = ['pm25', 'pm10', 'no2', 'o3', 'so2', 'co', 
                'temperature', 'humidity', 'pressure', 'wind_speed', 
                'wind_deg', 'clouds', 'hour', 'day_of_week', 'month']
    
    X = df[features]
    y = df['aqi_category']
    
    # Encode target
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    # Define models to experiment with
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
        "Ridge": RidgeClassifier(random_state=42)
    }
    
    best_f1 = 0
    best_model_name = ""
    all_metrics = []
    
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Metrics
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            print(f"{name} - Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")
            
            # Log params and metrics
            mlflow.log_param("model_type", name)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)
            
            # Log model
            # Log model and register it
            # Note: For Ridge, we use sklearn.log_model as it's a scikit-learn estimator
            if name == "RandomForest":
                mlflow.sklearn.log_model(model, "model", registered_model_name="RandomForest_AQI")
            elif name == "XGBoost":
                mlflow.xgboost.log_model(model, "model", registered_model_name="XGBoost_AQI")
            else:
                mlflow.sklearn.log_model(model, "model", registered_model_name="Ridge_AQI")
            
            # Accumulate metrics
            all_metrics.append({
                "model": name,
                "accuracy": acc,
                "f1_score": f1
            })

            if f1 > best_f1:
                best_f1 = f1
                best_model_name = name
                # Save locally for dashbord
                import pickle
                with open("best_model.pkl", "wb") as f:
                    pickle.dump(model, f)
                with open("label_encoder.pkl", "wb") as f:
                    pickle.dump(label_encoder, f)
    
    # Save all metrics
    with open("metrics.json", "w") as f:
        json.dump(all_metrics, f)

    print(f"Training complete. Best model: {best_model_name} with F1: {best_f1:.4f}")

if __name__ == "__main__":
    train_models()
