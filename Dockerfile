# 1. Use an official Python 3.10 image as the base
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the requirements file in first (for caching)
COPY requirements.txt .

# 4. Install the dependencies
# We use quotes just in case, to handle the [standard] part
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy all your project files into the container
# This includes main.py, phishing_model.joblib, and model_columns.joblib
COPY . .

# 6. Expose the port the app will run on
EXPOSE 8000

# 7. Define the command to run the app when the container starts
#CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["uvicorn", "main-email:app", "--host", "0.0.0.0", "--port", "8000"]
