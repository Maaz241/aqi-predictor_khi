import os
import sys

# Run feature pipeline
print("Running feature pipeline...")
os.system("python src/feature_engineering/feature_pipeline.py")

# Run training pipeline
print("Running training pipeline...")
os.system("python src/models/train.py")

# Run Streamlit dashboard
print("Running Streamlit dashboard...")
os.system("streamlit run src/dashboard/app.py")
