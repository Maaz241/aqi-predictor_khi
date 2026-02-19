import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score
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

# Prevent Windows cp1252 console errors when dependencies print Unicode characters.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.database.mongodb_client import MongoDBClient
from src.feature_engineering.feature_processor import FeatureProcessor

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


def _prepare_training_dataframe(df):
    base_features = [
        'pm25', 'pm10', 'no2', 'o3', 'so2', 'co',
        'temperature', 'humidity', 'pressure', 'wind_speed',
        'wind_deg', 'clouds'
    ]
    temporal_features = ['hour', 'day_of_week', 'month']
    features = base_features + temporal_features

    required_columns = base_features + ['aqi', 'timestamp']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns for training: {missing_columns}")

    training_df = df[required_columns].copy()

    numeric_columns = base_features + ['aqi']
    training_df[numeric_columns] = training_df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Parse and sort timestamp for a stable chronological split.
    if not pd.api.types.is_datetime64_any_dtype(training_df['timestamp']):
        numeric_ts = pd.to_numeric(training_df['timestamp'], errors='coerce')
        if not numeric_ts.isna().any():
            unit = 'ms' if numeric_ts.iloc[0] > 1e11 else 's'
            training_df['timestamp'] = pd.to_datetime(numeric_ts, unit=unit, errors='coerce')
        else:
            training_df['timestamp'] = pd.to_datetime(training_df['timestamp'], errors='coerce')

    training_df = training_df.sort_values('timestamp').reset_index(drop=True)

    # Recreate temporal features to avoid relying on stored columns.
    training_df['hour'] = training_df['timestamp'].dt.hour
    training_df['day_of_week'] = training_df['timestamp'].dt.dayofweek
    training_df['month'] = training_df['timestamp'].dt.month

    # Always recalculate labels from raw AQI so old/new rows use one schema.
    training_df['aqi_category'] = training_df['aqi'].apply(FeatureProcessor.get_aqi_category)

    training_df = training_df.dropna(subset=numeric_columns + temporal_features + ['timestamp', 'aqi_category'])
    if training_df.empty:
        raise ValueError("No valid training rows after preprocessing.")

    return training_df, features


def _split_chronological(X, y, train_fraction=0.8):
    split_idx = int(len(X) * train_fraction)
    if split_idx <= 0 or split_idx >= len(X):
        raise ValueError("Not enough data to create train/test splits.")
    return (
        X.iloc[:split_idx],
        X.iloc[split_idx:],
        y.iloc[:split_idx],
        y.iloc[split_idx:]
    )


def _evaluate_classification(model, X, y):
    X_train, X_test, y_train, y_test = _split_chronological(X, y)

    label_encoder = LabelEncoder()
    label_encoder.fit(y)

    y_train_encoded = label_encoder.transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    classes_only_in_test = sorted(set(y_test.unique()) - set(y_train.unique()))
    if classes_only_in_test:
        print(f"WARNING: Classes only in test split: {classes_only_in_test}")

    model.fit(X_train, y_train_encoded)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test_encoded, y_pred)
    f1 = f1_score(y_test_encoded, y_pred, average='weighted')

    return {
        "accuracy": acc,
        "f1_score": f1,
        "label_encoder": label_encoder,
        "X_train": X_train,
        "X_test": X_test
    }


def _evaluate_forecast_proxy(model, X, y, horizon_steps):
    # Realistic proxy: use features at time t to predict AQI category at t+horizon.
    if len(X) <= horizon_steps + 5:
        return None

    X_horizon = X.iloc[:-horizon_steps].reset_index(drop=True)
    y_horizon = y.iloc[horizon_steps:].reset_index(drop=True)

    try:
        result = _evaluate_classification(model, X_horizon, y_horizon)
    except ValueError:
        return None

    return {
        "accuracy": result["accuracy"],
        "f1_score": result["f1_score"],
        "samples": len(X_horizon)
    }


def _evaluate_transition_forecast(model, X, y, horizon_steps, train_fraction=0.8):
    # Model-based transition forecast:
    # 1) model predicts class at time t
    # 2) learned transition map converts predicted class(t) -> class(t+horizon)
    if len(y) <= horizon_steps + 5:
        return None

    X_t = X.iloc[:-horizon_steps].reset_index(drop=True)
    y_t = y.iloc[:-horizon_steps].reset_index(drop=True)
    y_future = y.iloc[horizon_steps:].reset_index(drop=True)

    split_idx = int(len(X_t) * train_fraction)
    if split_idx <= 0 or split_idx >= len(X_t):
        return None

    X_train, X_test = X_t.iloc[:split_idx], X_t.iloc[split_idx:]
    y_train_t = y_t.iloc[:split_idx]
    y_train_future = y_future.iloc[:split_idx]
    y_test_future = y_future.iloc[split_idx:]

    if y_train_t.empty or y_train_future.empty or y_test_future.empty:
        return None

    label_encoder = LabelEncoder()
    label_encoder.fit(y_t)

    y_train_t_encoded = label_encoder.transform(y_train_t)
    model.fit(X_train, y_train_t_encoded)

    pred_train_t = label_encoder.inverse_transform(model.predict(X_train))
    pred_test_t = label_encoder.inverse_transform(model.predict(X_test))

    transition_counts = {}
    for pred_label, true_future in zip(pred_train_t, y_train_future):
        if pred_label not in transition_counts:
            transition_counts[pred_label] = {}
        transition_counts[pred_label][true_future] = transition_counts[pred_label].get(true_future, 0) + 1

    if not transition_counts:
        return None

    default_future = y_train_future.mode().iloc[0]
    transition_map = {
        label: max(counts, key=counts.get)
        for label, counts in transition_counts.items()
    }
    y_pred_future = [transition_map.get(pred_label, default_future) for pred_label in pred_test_t]

    return {
        "accuracy": accuracy_score(y_test_future, y_pred_future),
        "f1_score": f1_score(y_test_future, y_pred_future, average='weighted'),
        "samples": len(y_test_future)
    }


def train_models():
    print("Starting Training Pipeline...")

    # 1. Load Data from MongoDB
    db_client = MongoDBClient()
    data = db_client.fetch_all()
    db_client.close()

    if not data:
        print("No data found in MongoDB.")
        return

    raw_df = pd.DataFrame(data)

    try:
        df, features = _prepare_training_dataframe(raw_df)
    except ValueError as exc:
        print(f"Training aborted: {exc}")
        return

    X = df[features]
    y = df['aqi_category']
    horizon_steps = max(1, int(os.getenv("AQI_METRIC_HORIZON_STEPS", "1")))

    try:
        X_train, X_test, _, _ = _split_chronological(X, y)
    except ValueError as exc:
        print(f"Training aborted: {exc}")
        return

    print(f"Training rows: {len(X_train)}, Test rows: {len(X_test)}")
    print(f"Category distribution: {y.value_counts().to_dict()}")
    print(f"Forecast-proxy horizon: {horizon_steps} step(s)")

    # Define models to experiment with
    model_factories = {
        "RandomForest": lambda: RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "XGBoost": lambda: xgb.XGBClassifier(
            eval_metric='mlogloss',
            random_state=42,
            n_estimators=200,
            max_depth=7,
            learning_rate=0.07,
            subsample=0.9,
            colsample_bytree=1.0,
            min_child_weight=1,
            gamma=0.0,
            reg_alpha=0.3,
            reg_lambda=2.0,
        ),
        "Ridge": lambda: RidgeClassifier(random_state=42)
    }

    best_f1 = float('-inf')
    best_model = None
    best_model_name = ""
    best_acc = 0
    best_metric_type = "transition_forecast"
    best_nowcast_acc = None
    best_nowcast_f1 = None
    best_proxy_acc = None
    best_proxy_f1 = None
    best_transition_acc = None
    best_transition_f1 = None
    all_metrics = []
    
    # Start a parent run for the main training session
    with mlflow.start_run(run_name="Main_AQI_Training_Session"):
        for name, model_factory in model_factories.items():
            model = model_factory()
            print(f"Training {name} locally...")

            nowcast_eval = _evaluate_classification(model, X, y)
            nowcast_acc = nowcast_eval["accuracy"]
            nowcast_f1 = nowcast_eval["f1_score"]
            label_encoder = nowcast_eval["label_encoder"]

            proxy_model = model_factory()
            forecast_proxy_eval = _evaluate_forecast_proxy(proxy_model, X, y, horizon_steps)
            proxy_acc = None
            proxy_f1 = None
            if forecast_proxy_eval is not None:
                proxy_acc = forecast_proxy_eval["accuracy"]
                proxy_f1 = forecast_proxy_eval["f1_score"]

            transition_model = model_factory()
            transition_eval = _evaluate_transition_forecast(transition_model, X, y, horizon_steps)
            transition_acc = None
            transition_f1 = None
            if transition_eval is not None:
                transition_acc = transition_eval["accuracy"]
                transition_f1 = transition_eval["f1_score"]

            # Primary leaderboard metric: transition-forecast.
            # Fallback order: direct forecast-proxy -> nowcast.
            primary_acc = transition_acc
            primary_f1 = transition_f1
            metric_type = "transition_forecast"
            if primary_f1 is None:
                primary_acc = proxy_acc
                primary_f1 = proxy_f1
                metric_type = "forecast_proxy"
            if primary_f1 is None:
                primary_acc = nowcast_acc
                primary_f1 = nowcast_f1
                metric_type = "nowcast"
            
            print(f"{name} - Nowcast Accuracy: {nowcast_acc:.4f}, F1-Score: {nowcast_f1:.4f}")
            if proxy_acc is not None and proxy_f1 is not None:
                print(
                    f"{name} - Forecast-Proxy(+{horizon_steps}) Accuracy: "
                    f"{proxy_acc:.4f}, F1-Score: {proxy_f1:.4f}"
                )
            else:
                print(f"{name} - Forecast-Proxy(+{horizon_steps}) unavailable (insufficient data)")
            if transition_acc is not None and transition_f1 is not None:
                print(
                    f"{name} - Transition-Forecast(+{horizon_steps}) Accuracy: "
                    f"{transition_acc:.4f}, F1-Score: {transition_f1:.4f}"
                )
            else:
                print(f"{name} - Transition-Forecast(+{horizon_steps}) unavailable (insufficient data)")
            print(f"{name} - Primary ({metric_type}) Accuracy: {primary_acc:.4f}, F1-Score: {primary_f1:.4f}")
            
            # Accumulate metrics for local storage
            all_metrics.append({
                "model": name,
                "accuracy": primary_acc,
                "f1_score": primary_f1,
                "metric_type": metric_type,
                "forecast_horizon_steps": horizon_steps,
                "nowcast_accuracy": nowcast_acc,
                "nowcast_f1_score": nowcast_f1,
                "forecast_proxy_accuracy": proxy_acc,
                "forecast_proxy_f1_score": proxy_f1,
                "transition_forecast_accuracy": transition_acc,
                "transition_forecast_f1_score": transition_f1
            })

            if primary_f1 > best_f1:
                best_f1 = primary_f1
                best_acc = primary_acc
                best_model = model
                best_model_name = name
                best_metric_type = metric_type
                best_nowcast_acc = nowcast_acc
                best_nowcast_f1 = nowcast_f1
                best_proxy_acc = proxy_acc
                best_proxy_f1 = proxy_f1
                best_transition_acc = transition_acc
                best_transition_f1 = transition_f1

        # 3. Log ONLY the best model to MLflow (Nested under parent run)
        if best_model:
            with mlflow.start_run(run_name=f"Best_Model_{best_model_name}", nested=True):
                print(f"Logging best model ({best_model_name}) to MLflow...")
                mlflow.log_param("model_type", best_model_name)
                mlflow.log_param("primary_metric_type", best_metric_type)
                mlflow.log_param("forecast_horizon_steps", horizon_steps)
                mlflow.log_metric("accuracy", best_acc)
                mlflow.log_metric("f1_score", best_f1)
                if best_nowcast_acc is not None:
                    mlflow.log_metric("nowcast_accuracy", best_nowcast_acc)
                if best_nowcast_f1 is not None:
                    mlflow.log_metric("nowcast_f1_score", best_nowcast_f1)
                if best_proxy_acc is not None:
                    mlflow.log_metric("forecast_proxy_accuracy", best_proxy_acc)
                if best_proxy_f1 is not None:
                    mlflow.log_metric("forecast_proxy_f1_score", best_proxy_f1)
                if best_transition_acc is not None:
                    mlflow.log_metric("transition_forecast_accuracy", best_transition_acc)
                if best_transition_f1 is not None:
                    mlflow.log_metric("transition_forecast_f1_score", best_transition_f1)

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

    print(
        f"Training complete. Best model: {best_model_name} "
        f"({best_metric_type}) with F1: {best_f1:.4f}"
    )


if __name__ == "__main__":
    train_models()
