Hybrid Phishing Detection System

Email and Website Based Phishing Classification
Version 2.0

Overview

This project builds and serves a hybrid phishing detection model that analyzes two types of inputs:

Website-based numerical features (UCI Phishing Websites Dataset)

Email text–based phishing indicators (Phishing_Email.csv)

The system combines both datasets into one unified training pipeline and produces a single machine-learning model that can detect phishing attempts from either source.

The project includes:

A hybrid training script (train-hybrid.py)

FastAPI backend for prediction (main_email.py)

Dockerized API

CI/CD pipeline using GitHub Actions (ci.yml)

1. Project Structure

phishing-backend/
│
├── main_email.py               # FastAPI application
├── train-hybrid.py             # Hybrid model training script
├── phishing_model.joblib       # Trained model (generated)
├── model_columns.joblib        # Feature column order for prediction
├── email_scaler.joblib         # Scaler for email features
├── prediction_log.csv          # Saved logs of predictions
├── requirements.txt
├── Dockerfile
└── .github/workflows/ci.yml    # GitHub Actions pipeline

2. What Each File Does
main_email.py (FastAPI App)

This file runs the API that accepts inputs and returns phishing predictions.

Key features:

Two prediction endpoints:

/predict/website

/predict/email

Extracts 20+ handcrafted email features

Applies scaling and converts to -1, 0, 1 buckets

Reorders features exactly as the model was trained

Generates predictions with confidence scores

Logs all predictions into prediction_log.csv

Provides Swagger docs at /docs

It does not include:

Drift detection

Kafka streaming

Feedback training loop

You send a JSON request and get a clean JSON response.



train-hybrid.py (Hybrid Training Pipeline)

This script trains the hybrid phishing model using two datasets:

UCI Website Dataset (numerical)

Phishing Email Dataset (text → numerical features)

Major steps:

Load UCI dataset using ucimlrepo

Load phishing_email.csv from data/

Extract features from email text (length, URLs, urgency, uppercase ratio, keywords, etc.)

Normalize email features using StandardScaler

Convert values into -1, 0, 1 buckets to match UCI dataset scale

Add a source_website indicator column

Combine both datasets into one hybrid dataset

Train three ML models:

Logistic Regression

Random Forest

Gradient Boosting

Select best model based on accuracy

Save:

phishing_model.joblib

model_columns.joblib

email_scaler.joblib

The training script originally used MLflow, but MLflow was removed from Docker because it caused installation failures. It is still present for GitHub Actions usage only.



.github/workflows/ci.yml (CI/CD Pipeline)

This file creates a full automation pipeline that runs on every push to main.

Pipeline steps:

Checkout repository

Print folder debug tree

Install dependencies

Run pytest tests

Download phishing_email.csv from GitHub Releases

Run train-hybrid.py

Upload the trained model as an artifact

Build Docker Image

Login to Docker Hub

Push image to Docker Hub

Notes:

mlflow must remain in requirements.txt for GitHub Actions to work.

mlflow must be removed when building Docker locally.

To run the CI/CD pipelines , run these commands:
git add .
git commit -m "update"
git push origin main

Once this is done Github reads the .yml file and triggers the CI/CD pipeline steps



3. Commands Used During Development
Running tests

pytest -v

Run backend locally 

uvicorn main_email:app --reload --port 8000

Rebuilding Docker image

docker build -t joteena/phishing-api:latest .

Running Docker container

docker run -p 8000:8000 joteena/phishing-api:latest

Sending request to Docker

curl -X POST "http://localhost:8000/predict/email" \
     -H "Content-Type: application/json" \
     -d '{"email_text": "Congratulations, you won a free iPhone."}'


4. GitHub Actions Notes

To make GitHub Actions succeed:

Keep mlflow in requirements.txt

The pipeline downloads the dataset from:
https://github.com/T490-hue/phishing-backend/releases/download/v1.0.0/phishing_email.csv


5. Running the API

Once Docker container starts:

Swagger documentation is available at
http://localhost:8000/docs


6. Output Example

POST /predict/email
Input:

{
  "email_text": "Your account is suspended. Click here to verify."
}


Output:

{
  "prediction": "Phishing",
  "confidence": "0.9821",
  "is_phishing": 1,
  "input_type": "email",
  "analysis": {
      "suspicious_keywords_found": 4,
      "urls_found": 1,
      "urgency_indicators": 1,
      "financial_terms": 0,
      "has_generic_greeting": false
  }
}
7. Summary

You built a full hybrid phishing detection system that:

Combines two datasets (website + email)

Extracts text-based phishing indicators

Normalizes email features to match UCI dataset

Trains a unified machine learning model

Deploys using FastAPI

Automates training and Docker image creation using GitHub Actions

Serves predictions through a Dockerized API
