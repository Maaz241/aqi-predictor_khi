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

# Check for secrets
if not dagshub_token:
    print("WARNING: DAGSHUB_TOKEN not found in environment variables!")
if not repo_owner:
    print("WARNING: DAGSHUB_REPO_OWNER not found in environment variables!")

# Authentication Logic
if os.getenv("MLFLOW_TRACKING_URI"):
    # CI/CD or Manual Remote Run
    print(f"MLFLOW_TRACKING_URI found: {os.getenv('MLFLOW_TRACKING_URI')}")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    
    # Set credentials for MLflow authentication
    if dagshub_token and repo_owner:
        os.environ['MLFLOW_TRACKING_USERNAME'] = repo_owner
        os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token
        print("MLflow credentials set from environment variables.")
    else:
        print("ERROR: DagsHub credentials missing for remote tracking!")
else:
    # Local Interactive Run
    if dagshub_token and repo_owner and repo_name:
        print("Initializing DagsHub (Local Run)...")
        dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)

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
    best_model = None
    best_model_name = ""
    best_acc = 0
    all_metrics = []
    
    # Start a parent run for the main training session
    with mlflow.start_run(run_name="Main_AQI_Training_Session"):
        for name, model in models.items():
            print(f"Training {name} locally...")
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Metrics
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            print(f"{name} - Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")
            
            # Accumulate metrics for local storage
            all_metrics.append({
                "model": name,
                "accuracy": acc,
                "f1_score": f1
            })

            if f1 > best_f1:
                best_f1 = f1
                best_acc = acc
                best_model = model
                best_model_name = name
        
        # 3. Log ONLY the best model to MLflow (Nested under parent run)
        if best_model:
            with mlflow.start_run(run_name=f"Best_Model_{best_model_name}", nested=True):
                print(f"Logging best model ({best_model_name}) to MLflow...")
                mlflow.log_param("model_type", best_model_name)
                mlflow.log_metric("accuracy", best_acc)
                mlflow.log_metric("f1_score", best_f1)
                
                # Log model parameters
                if hasattr(best_model, 'get_params'):
                    mlflow.log_params(best_model.get_params())
                
                # Register and log the model
                if best_model_name == "RandomForest":
                    mlflow.sklearn.log_model(best_model, "model", registered_model_name="RandomForest_AQI")
                elif best_model_name == "XGBoost":
                    mlflow.xgboost.log_model(best_model, "model", registered_model_name="XGBoost_AQI")
                else:
                    mlflow.sklearn.log_model(best_model, "model", registered_model_name="Ridge_AQI")
                
                run_id = mlflow.active_run().info.run_id
                print(f"Logged best model to MLflow. Run ID: {run_id}")

            # Save locally for dashboard
            import pickle
            with open("best_model.pkl", "wb") as f:
                pickle.dump(best_model, f)
            with open("label_encoder.pkl", "wb") as f:
                pickle.dump(label_encoder, f)
    
    # Save all metrics locally
    with open("metrics.json", "w") as f:
        json.dump(all_metrics, f)

    print(f"Training complete. Best model: {best_model_name} with F1: {best_f1:.4f}")

if __name__ == "__main__":
    train_models()
