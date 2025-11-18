from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import atexit
from csv import DictWriter
import os
import re
import numpy as np

# Create FastAPI app
app = FastAPI(title="Hybrid Phishing Detection API", version="2.0")

# CORS
origins = ["http://localhost:3000", "http://localhost:5173"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and resources
try:
    model = joblib.load("phishing_model.joblib")
    model_columns = joblib.load("model_columns.joblib")
    email_scaler = joblib.load("email_scaler.joblib")
    print("✓ Model, columns, and scaler loaded successfully")
except FileNotFoundError as e:
    print(f"✗ Error loading files: {e}")
    print("  Please run train_hybrid_email.py first")
    model = None

# Input schemas
class WebsiteFeatures(BaseModel):
    """Original website-based features"""
    features: dict

class EmailFeatures(BaseModel):
    """Email text for analysis"""
    email_text: str
    subject: str = ""

# Logging setup
LOG_FILE = 'prediction_log.csv'
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w', newline='') as f:
        writer = DictWriter(f, fieldnames=model_columns + ['prediction', 'confidence', 'input_type'])
        writer.writeheader()

log_file_writer = open(LOG_FILE, 'a', newline='')
log_writer = DictWriter(log_file_writer, fieldnames=model_columns + ['prediction', 'confidence', 'input_type'])

@atexit.register
def close_log_file():
    print("Closing log file...")
    log_file_writer.close()

# Helper function: Extract email features
def extract_email_features(text, subject=''):
    """Extract features from email text"""
    combined_text = f"{subject} {text}".lower()
    
    features = {}
    
    # Length-based
    features['email_length'] = len(combined_text)
    features['word_count'] = len(combined_text.split())
    features['avg_word_length'] = np.mean([len(w) for w in combined_text.split()]) if combined_text.split() else 0
    
    # Suspicious keywords
    phishing_keywords = [
        'urgent', 'verify', 'account', 'suspended', 'click', 'confirm',
        'password', 'credit', 'bank', 'security', 'update', 'expire',
        'winner', 'prize', 'congratulations', 'claim', 'free', 'offer'
    ]
    features['suspicious_keywords'] = sum(kw in combined_text for kw in phishing_keywords)
    
    # URLs
    features['has_url'] = 1 if re.search(r'http[s]?://', combined_text) else 0
    features['url_count'] = len(re.findall(r'http[s]?://', combined_text))
    
    # Special characters
    features['exclamation_count'] = combined_text.count('!')
    features['question_count'] = combined_text.count('?')
    features['dollar_sign'] = combined_text.count('$')
    features['at_symbol'] = combined_text.count('@')
    
    # Uppercase ratio
    features['uppercase_ratio'] = sum(1 for c in combined_text if c.isupper()) / len(combined_text) if combined_text else 0
    
    # Numbers
    features['has_numbers'] = 1 if re.search(r'\d', combined_text) else 0
    features['number_count'] = len(re.findall(r'\d+', combined_text))
    
    # Email addresses
    features['email_addresses'] = len(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', combined_text))
    
    # Urgency
    urgent_words = ['urgent', 'immediate', 'action required', 'act now', 'limited time']
    features['urgency_score'] = sum(word in combined_text for word in urgent_words)
    
    # Financial terms
    financial_terms = ['money', 'payment', 'transfer', 'account', 'bank', 'credit card']
    features['financial_terms'] = sum(term in combined_text for term in financial_terms)
    
    # Generic greeting
    generic_greetings = ['dear customer', 'dear user', 'dear member', 'valued customer']
    features['generic_greeting'] = 1 if any(g in combined_text for g in generic_greetings) else 0
    
    # Source indicator
    features['source_website'] = 0  # Email input
    
    return features

# Endpoints
@app.get("/")
def read_root():
    return {
        "message": "Hybrid Phishing Detection API",
        "version": "2.0",
        "features": ["Website Features", "Email Text Analysis"],
        "model_loaded": model is not None
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "supported_inputs": ["website_features", "email_text"]
    }

@app.post("/predict/website")
def predict_website(data: WebsiteFeatures):
    """Predict based on website features (original UCI format)"""
    if model is None:
        return {"error": "Model not loaded"}
    
    try:
        input_df = pd.DataFrame(data.features, index=[0])
        input_df.columns = input_df.columns.str.lower()
        
        # Add source indicator
        input_df['source_website'] = 1
        
        # Reorder columns
        input_df = input_df.reindex(columns=model_columns, fill_value=0)
        
        # Predict
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)
        
        result = "Phishing" if prediction[0] == 1 else "Safe"
        confidence = probability[0][prediction[0]]
        
        # Log
        log_entry = {col: input_df[col].values[0] if col in input_df.columns else 0 
                     for col in model_columns}
        log_entry['prediction'] = result
        log_entry['confidence'] = f"{confidence:.4f}"
        log_entry['input_type'] = 'website'
        
        try:
            log_writer.writerow(log_entry)
            log_file_writer.flush()
        except Exception as e:
            print(f"Log error: {e}")
        
        return {
            "prediction": result,
            "confidence": f"{confidence:.4f}",
            "is_phishing": int(prediction[0]),
            "input_type": "website"
        }
    
    except Exception as e:
        return {"error": f"Prediction error: {str(e)}"}

@app.post("/predict/email")
def predict_email(data: EmailFeatures):
    """Predict based on email text"""
    if model is None:
        return {"error": "Model not loaded"}
    
    try:
        # Extract features
        email_features = extract_email_features(data.email_text, data.subject)
        
        # Create DataFrame
        input_df = pd.DataFrame([email_features])
        
        # Normalize
        feature_cols = [col for col in input_df.columns if col != 'source_website']
        input_df[feature_cols] = email_scaler.transform(input_df[feature_cols])
        
        # Convert to -1, 0, 1 scale
        for col in feature_cols:
            input_df[col] = pd.cut(
                input_df[col],
                bins=[-np.inf, -0.5, 0.5, np.inf],
                labels=[-1, 0, 1]
            ).astype(int)
        
        # Reorder columns
        input_df = input_df.reindex(columns=model_columns, fill_value=0)
        
        # Predict
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)
        
        result = "Phishing" if prediction[0] == 1 else "Safe"
        confidence = probability[0][prediction[0]]
        
        # Log
        log_entry = {col: input_df[col].values[0] if col in input_df.columns else 0 
                     for col in model_columns}
        log_entry['prediction'] = result
        log_entry['confidence'] = f"{confidence:.4f}"
        log_entry['input_type'] = 'email'
        
        try:
            log_writer.writerow(log_entry)
            log_file_writer.flush()
        except Exception as e:
            print(f"Log error: {e}")
        
        # Return with analysis
        return {
            "prediction": result,
            "confidence": f"{confidence:.4f}",
            "is_phishing": int(prediction[0]),
            "input_type": "email",
            "analysis": {
                "suspicious_keywords_found": email_features['suspicious_keywords'],
                "urls_found": email_features['url_count'],
                "urgency_indicators": email_features['urgency_score'],
                "financial_terms": email_features['financial_terms'],
                "has_generic_greeting": bool(email_features['generic_greeting'])
            }
        }
    
    except Exception as e:
        return {"error": f"Prediction error: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
