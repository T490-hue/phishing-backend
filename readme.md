# Phishing Detection MLOps Backend

This repository contains the complete MLOps backend for our Phishing Detection System. It's responsible for training the model, serving it as an API, and monitoring it for drift.

This system is built with **FastAPI**, **Docker**, and **MLflow**.

---

## 🚀 What Each File Does

* **`train.py`**: (Modules 1-4, 12) Loads the UCI dataset, trains the `LogisticRegression` model, logs it with MLflow, and saves `phishing_model.joblib` and `model_columns.joblib`.
* **`main.py`**: (Modules 5, 7) The FastAPI server. It loads the model, handles CORS, and creates the `/predict` endpoint. It also logs every prediction to `prediction_log.csv`.
* **`check_drift.py`**: (Module 11) The monitoring script. It compares `prediction_log.csv` against the original training data to detect model drift.
* **`Dockerfile`**: (Module 6) The blueprint to package the entire FastAPI application into a portable Docker container.
* **`requirements.txt`**: A list of all Python libraries needed for the project.

---

## ⚙️ How to Run This Project (Step-by-Step)

Follow these steps exactly to get the backend server running.

### 1. Set Up the Environment

First, clone the repository and set up the Python virtual environment.

```bash
# Clone the repo (if you haven't)
# git clone ...

# Create a virtual environment
python -m venv .venv

# Activate it (on Mac/Linux)
source .venv/bin/activate
# (on Windows, use: .\.venv\Scripts\activate)

# Install all required libraries
pip install -r requirements.txt

You're right, a README.md file is the perfect way to do this. This is the most important file for team collaboration.

Here are two complete README.md files, one for each of your new repositories. Just copy and paste this text into a new file named README.md in each project.

1. For phishing-backend (Your Python Project)
Create a new file named README.md in your phishing-mlops-project folder and paste this in:

Markdown

# Phishing Detection MLOps Backend

This repository contains the complete MLOps backend for our Phishing Detection System. It's responsible for training the model, serving it as an API, and monitoring it for drift.

This system is built with **FastAPI**, **Docker**, and **MLflow**.

---

## 🚀 What Each File Does

* **`train.py`**: (Modules 1-4, 12) Loads the UCI dataset, trains the `LogisticRegression` model, logs it with MLflow, and saves `phishing_model.joblib` and `model_columns.joblib`.
* **`main.py`**: (Modules 5, 7) The FastAPI server. It loads the model, handles CORS, and creates the `/predict` endpoint. It also logs every prediction to `prediction_log.csv`.
* **`check_drift.py`**: (Module 11) The monitoring script. It compares `prediction_log.csv` against the original training data to detect model drift.
* **`Dockerfile`**: (Module 6) The blueprint to package the entire FastAPI application into a portable Docker container.
* **`requirements.txt`**: A list of all Python libraries needed for the project.

---

## ⚙️ How to Run This Project (Step-by-Step)

Follow these steps exactly to get the backend server running.

### 1. Set Up the Environment

First, clone the repository and set up the Python virtual environment.

```bash
# Clone the repo (if you haven't)
# git clone ...

# Create a virtual environment
python -m venv .venv

# Activate it (on Mac/Linux)
source .venv/bin/activate
# (on Windows, use: .\.venv\Scripts\activate)

# Install all required libraries
pip install -r requirements.txt

2. Train the Model
You must run the training script first. This creates the phishing_model.joblib and model_columns.joblib files that the API needs.

python train.py

3. Build the Docker Container
Make sure you have Docker Desktop installed and running. This command builds your application "image".

docker build -t phishing-api .

4. Run the API Server
This command starts your API, which will run at http://127.0.0.1:8000. This container must be running for the frontend to work.

docker run -p 8000:8000 phishing-api

---

## 🔬 How to Check for Model Drift (Module 7 & 11)

This project logs every prediction to a file *inside* the Docker container. You can run a script to check if this new data has "drifted" from the original training data.

1.  Run the API: Make sure your Docker container is running.
    
    docker run -p 8000:8000 phishing-api

2.  Generate Log Data: Go to the frontend website (at `http://localhost:3000`) and click the test buttons 5-10 times to generate some prediction logs.

3.  Find Container ID: Open a new terminal and find the ID of your running container.
    
    docker ps
  
    (Look for the ID next to the `phishing-api` name).

4.  Copy the Log File Out: Use this command to copy the log file from the container to your computer. Replace `YOUR_CONTAINER_ID` with the ID you just found.
    
    docker cp YOUR_CONTAINER_ID:/app/prediction_log.csv .

5.  Run the Drift Script: Now that the file is on your computer, run the drift check script.
    ```bash
    python check_drift.py
    ```

You will see a report in your terminal that compares the statistics of the new data to the original data and warns you if drift is detected.
