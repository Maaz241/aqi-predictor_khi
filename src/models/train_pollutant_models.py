import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import dagshub
from dotenv import load_dotenv

from src.database.mongodb_client import MongoDBClient

# Load environment variables
load_dotenv()

# DagsHub/MLflow setup
dagshub_token = os.getenv("DAGSHUB_TOKEN")
repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
repo_name = os.getenv("DAGSHUB_REPO_NAME")

if os.getenv("MLFLOW_TRACKING_URI"):
    print(f"MLFLOW_TRACKING_URI found: {os.getenv('MLFLOW_TRACKING_URI')}")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    
    if dagshub_token and repo_owner:
        os.environ['MLFLOW_TRACKING_USERNAME'] = repo_owner
        os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token
        print("MLflow credentials set from environment variables.")
else:
    if dagshub_token and repo_owner and repo_name:
        print("Initializing DagsHub (Local Run)...")
        dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)


def create_lag_features(df, pollutants, n_lags=3):
    """Create lag features for pollutants"""
    df = df.copy()
    
    for pollutant in pollutants:
        for lag in range(1, n_lags + 1):
            df[f'{pollutant}_lag{lag}'] = df[pollutant].shift(lag)
    
    return df


def create_rolling_features(df, pollutants, window=6):
    """Create rolling statistics (6 intervals = 24 hours)"""
    df = df.copy()
    
    for pollutant in pollutants:
        df[f'{pollutant}_rolling_mean'] = df[pollutant].rolling(window=window).mean()
        df[f'{pollutant}_rolling_std'] = df[pollutant].rolling(window=window).std()
    
    return df


def prepare_data():
    """Fetch and prepare data from MongoDB"""
    print("Fetching data from MongoDB...")
    client = MongoDBClient()
    data = list(client.collection.find().sort('timestamp', 1))
    client.close()
    
    df = pd.DataFrame(data)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"Loaded {len(df)} records")
    
    # Define pollutants
    pollutants = ['pm25', 'pm10', 'no2', 'o3', 'so2', 'co']
    
    # Create lag features
    print("Creating lag features...")
    df = create_lag_features(df, pollutants, n_lags=3)
    
    # Create rolling features
    print("Creating rolling statistics...")
    df = create_rolling_features(df, pollutants, window=6)
    
    # Drop rows with NaN (due to lag/rolling)
    df = df.dropna().reset_index(drop=True)
    
    print(f"After feature engineering: {len(df)} records")
    
    return df, pollutants


def train_pollutant_models(df, pollutants):
    """Train regression models for each pollutant"""
    
    # Feature columns (excluding target pollutants themselves)
    weather_features = ['temperature', 'humidity', 'pressure', 'wind_speed', 'wind_deg', 'clouds']
    temporal_features = ['hour', 'day_of_week', 'month']
    
    models = {}
    metrics_summary = []
    
    mlflow.set_experiment("pollutant_prediction")
    
    # Start a parent run for this training session
    with mlflow.start_run(run_name=f"Pollutant_Training_Session"):
        
        for pollutant in pollutants:
            print(f"\n{'='*60}")
            print(f"Training models for {pollutant.upper()}")
            print(f"{'='*60}")
            
            # Build feature list for this pollutant
            lag_features = [f'{pollutant}_lag{i}' for i in range(1, 4)]
            rolling_features = [f'{pollutant}_rolling_mean', f'{pollutant}_rolling_std']
            
            # Include lag features from OTHER pollutants as well (cross-pollutant effects)
            other_pollutants = [p for p in pollutants if p != pollutant]
            other_lag_features = []
            for other_p in other_pollutants:
                other_lag_features.append(f'{other_p}_lag1')  # Only lag1 from others
            
            feature_cols = lag_features + rolling_features + other_lag_features + weather_features + temporal_features
            
            X = df[feature_cols]
            y = df[pollutant]
            
            # Train/test split (80/20)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
            
            print(f"Training set: {len(X_train)}, Test set: {len(X_test)}")
            
            # Try multiple models - evaluate locally first
            model_configs = {
                'XGBoost': XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
                'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
                'Ridge': Ridge(alpha=1.0)
            }
            
            best_model = None
            best_rmse = float('inf')
            best_mae = float('inf')
            best_r2 = float('-inf')
            best_model_name = None
            
            # 1. Evaluate all models locally
            for model_name, model in model_configs.items():
                # Train
                model.fit(X_train, y_train)
                
                # Predict
                y_pred = model.predict(X_test)
                
                # Metrics
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                print(f"\n{model_name}:")
                print(f"  RMSE: {rmse:.4f}")
                print(f"  MAE:  {mae:.4f}")
                print(f"  R²:   {r2:.4f}")
                
                # Track best model
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_mae = mae
                    best_r2 = r2
                    best_model = model
                    best_model_name = model_name
            
            # 2. Log ONLY the best model to MLflow (Nested under parent run)
            with mlflow.start_run(run_name=f"{pollutant}_{best_model_name}", nested=True):
                mlflow.log_param("pollutant", pollutant)
                mlflow.log_param("model_type", best_model_name)
                mlflow.log_param("n_features", len(feature_cols))
                mlflow.log_metric("rmse", best_rmse)
                mlflow.log_metric("mae", best_mae)
                mlflow.log_metric("r2", best_r2)
                
                # Log model parameters if available
                if hasattr(best_model, 'get_params'):
                    mlflow.log_params(best_model.get_params())
                
                # Retrieve the run ID to print it
                run_id = mlflow.active_run().info.run_id
                print(f"Logged best model ({best_model_name}) to MLflow. Run ID: {run_id}")
            
            # Store best model for saving
            models[pollutant] = {
                'model': best_model,
                'features': feature_cols,
                'model_name': best_model_name,
                'rmse': best_rmse
            }
            
            metrics_summary.append({
                'pollutant': pollutant,
                'best_model': best_model_name,
                'rmse': best_rmse
            })
            
            print(f"\n✓ Best model for {pollutant.upper()}: {best_model_name} (RMSE: {best_rmse:.4f})")
    
    return models, metrics_summary


def save_models(models):
    """Save pollutant models to disk"""
    output_path = "pollutant_models.pkl"
    
    with open(output_path, 'wb') as f:
        pickle.dump(models, f)
    
    print(f"\n✓ Saved pollutant models to {output_path}")


if __name__ == "__main__":
    print("Starting Pollutant Prediction Model Training...")
    print("="*60)
    
    # Prepare data
    df, pollutants = prepare_data()
    
    # Train models
    models, metrics_summary = train_pollutant_models(df, pollutants)
    
    # Save models
    save_models(models)
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    summary_df = pd.DataFrame(metrics_summary)
    print(summary_df.to_string(index=False))
    print("\n✓ Training complete!")
