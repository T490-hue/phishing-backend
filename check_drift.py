import pandas as pd
from ucimlrepo import fetch_ucirepo

print("--- Running Model Drift Detection Script ---")

# --- 1. Load Original Training Data ---
print("Loading original training data...")
try:
    phishing_websites = fetch_ucirepo(id=327)
    X_train_original = phishing_websites.data.features
    X_train_original.columns = X_train_original.columns.str.lower()
    print("Original data loaded.")
except Exception as e:
    print(f"Failed to load training data: {e}")
    exit()

# --- 2. Load Production Prediction Log ---
LOG_FILE = 'prediction_log.csv'
print(f"Loading production data from {LOG_FILE}...")
try:
    prod_data_df = pd.read_csv(LOG_FILE)
    # Drop the log-only columns to match training data
    prod_data_df = prod_data_df.drop(columns=['prediction', 'confidence'], errors='ignore')
    if prod_data_df.empty:
        print("Log file is empty. No data to check. Exiting.")
        exit()
    print(f"Loaded {len(prod_data_df)} predictions from log.")
except FileNotFoundError:
    print(f"Log file not found. No data to check. Exiting.")
    exit()

# --- 3. Check for Drift (Simple Statistical Check) ---
# We will compare the 'mean' of a few key features.

print("\n--- Checking for Data Drift ---")
drift_detected = False

# Define a simple threshold for drift (e.g., 10% change in mean)
DRIFT_THRESHOLD = 0.1 

features_to_check = ['url_length', 'sslfinal_state', 'page_rank', 'web_traffic']

for col in features_to_check:
    if col in X_train_original.columns and col in prod_data_df.columns:
        mean_original = X_train_original[col].mean()
        mean_prod = prod_data_df[col].mean()

        # Calculate percent change
        percent_change = abs(mean_prod - mean_original) / mean_original

        print(f"Feature '{col}': Original Mean={mean_original:.4f}, Production Mean={mean_prod:.4f}, Change={percent_change:.2%}")

        if percent_change > DRIFT_THRESHOLD:
            print(f"  ** DRIFT DETECTED! ** Feature '{col}' has changed by more than {DRIFT_THRESHOLD:.0%}")
            drift_detected = True

if not drift_detected:
    print("\n--- No significant data drift detected. ---")

print("--- Drift check complete. ---")