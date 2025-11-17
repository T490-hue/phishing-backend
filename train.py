import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from ucimlrepo import fetch_ucirepo
import joblib

print("--- Starting Model Training ---")

# --- Module 1 & 2: Data Acquisition & Preprocessing ---
# Fetch dataset (ID 327 is 'Phishing Websites')
try:
    phishing_websites = fetch_ucirepo(id=327)
    X = phishing_websites.data.features
    y = phishing_websites.data.targets
    print("Dataset loaded successfully from UCI.")
except Exception as e:
    print(f"Failed to load data: {e}")
    exit()

# Preprocessing: The target 'Result' is -1 (safe) and 1 (phishing).
# We will change it to 0 (safe) and 1 (phishing) for simplicity.
y = y.squeeze() # Convert from DataFrame to Series
y = y.replace(-1, 0)
X.columns = X.columns.str.lower()
model_columns = list(X.columns)
joblib.dump(model_columns, 'model_columns.joblib')

print(f"Data shape: {X.shape}")
print(f"Target distribution:\n{y.value_counts()}")

# --- Module 3: Model Development & Training ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Module 12: Experiment Tracking (MLflow) ---
# Set an experiment name
mlflow.set_experiment("Phishing-Detection-v1")

# Start an MLflow run
with mlflow.start_run() as run:
    print(f"MLflow Run ID: {run.info.run_id}")

    # Train a simple model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Model Accuracy: {acc:.4f}")

    # Log metrics and parameters
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_metric("accuracy", acc)

    # --- Module 4: Model Serialization ---
    # Log the model *with* MLflow (best practice)
    mlflow.sklearn.log_model(model, "model")

    # Also save a simple 'joblib' file for our API to use
    joblib.dump(model, 'phishing_model.joblib')
    print("Model saved to 'phishing_model.joblib'")

print("--- Model Training Finished ---")