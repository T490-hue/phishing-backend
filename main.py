from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn # We'll use this for a run command later
from fastapi.middleware.cors import CORSMiddleware
import atexit
from csv import DictWriter
import os

# 1. Create the FastAPI app
app = FastAPI(title="Phishing Detection API", version="1.0")

# Define the allowed origins (your React app's address)
origins = [
    "http://localhost:3000",
]

# Add the CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allow all headers
)
# 2. Load the trained model
try:
    model = joblib.load("phishing_model.joblib")
    model_columns = joblib.load("model_columns.joblib") # <-- ADD THIS
    print("Model and columns loaded successfully.")
except FileNotFoundError:
    print("Model file 'phishing_model.joblib' not found. Please run train.py first.")
    model = None

# 3. Define the input data format using Pydantic
# This ensures the data we receive is valid
# 3. Define the input data format using Pydantic
# 3. Define the input data format using Pydantic
class PhishingFeatures(BaseModel):
    # We use a 'dict' so we can pass all 30 features as a single JSON object
    features: dict
    
    # Add an example for the API docs
    class Config:
        json_schema_extra = {
            "example": {
                "features": {
                    "having_ip_address": 1,
                    "url_length": 1,
                    "shortining_service": 1,
                    "having_at_symbol": 1,
                    "double_slash_redirecting": 1,
                    "prefix_suffix": 1,
                    "having_sub_domain": 0,
                    "sslfinal_state": 1,
                    "domain_registration_length": 1,
                    "favicon": 1,
                    "port": 1,
                    "https_token": 1,
                    "request_url": 1,
                    "url_of_anchor": 0,
                    "links_in_tags": 0,
                    "sfh": 1,
                    "submitting_to_email": 1,
                    "abnormal_url": 1,
                    "redirect": 0,
                    "on_mouseover": 1,
                    "rightclick": 1,
                    "popupwindow": 1,
                    "iframe": 1,
                    "age_of_domain": 1,
                    "dnsrecord": 1,
                    "web_traffic": 0,
                    "page_rank": 1,
                    "google_index": 1,
                    "links_pointing_to_page": 0,
                    "statistical_report": 1
                }
            }
        }

# --- Logging Setup ---
# We will log prediction data to a CSV file
LOG_FILE = 'prediction_log.csv'

# Get the column names from the joblib file
try:
    LOG_COLUMNS = joblib.load("model_columns.joblib")
    # We also want to log the prediction and confidence
    LOG_COLUMNS.extend(['prediction', 'confidence'])
except FileNotFoundError:
    LOG_COLUMNS = ['prediction', 'confidence'] # Fallback

# Create the file and write the header if it doesn't exist
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w', newline='') as f:
        writer = DictWriter(f, fieldnames=LOG_COLUMNS)
        writer.writeheader()

# Open the file in append mode
log_file_writer = open(LOG_FILE, 'a', newline='')
log_writer = DictWriter(log_file_writer, fieldnames=LOG_COLUMNS)

# Add a function to close the file when the app exits
@atexit.register
def close_log_file():
    print("Closing log file...")
    log_file_writer.close()

# --- End Logging Setup ---        

# 4. Create the API endpoints

@app.get("/")
def read_root():
    return {"message": "Phishing Detection API is running."}

@app.post("/predict")
def predict_phishing(data: PhishingFeatures):
    if model is None:
        return {"error": "Model not loaded. Please check the server logs."}
        
    try:
        # Convert the input dictionary into a Pandas DataFrame
        # The model expects a DataFrame in the *exact* format it was trained on
        input_df = pd.DataFrame(data.features, index=[0])
        input_df.columns = input_df.columns.str.lower()
        
        input_df = input_df[model_columns]
        
        # Make the prediction
        prediction = model.predict(input_df)
        
        # Get the prediction probability (how confident it is)
        probability = model.predict_proba(input_df)
        
        # Format the result
        result = "Phishing" if prediction[0] == 1 else "Safe"
        confidence = probability[0][prediction[0]] # Get the prob for the predicted class
        
        # --- Module 7: Monitoring & Logging ---
        # Create a log entry
        log_entry = data.features.copy() # Start with all the input features
        log_entry['prediction'] = result
        log_entry['confidence'] = f"{confidence:.4f}"

        # Write to the CSV file
        try:
            log_writer.writerow(log_entry)
            log_file_writer.flush()
        except Exception as e:
            print(f"Error writing to log: {e}")
        # --- End Logging ---

        # Return the response to the user
        return {
            "prediction": result,
            "confidence": f"{confidence:.4f}",
            "is_phishing": int(prediction[0])
        }

    except Exception as e:
        return {"error": f"Error during prediction: {str(e)}"}

# 5. Add a 'main' block to run the app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)