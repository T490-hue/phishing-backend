from fastapi.testclient import TestClient
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main_email import app

client = TestClient(app)

def test_root_alive():
    """Check API root or health endpoint."""
    response = client.get("/")
    assert response.status_code in [200, 404]  # In case root not defined

def test_predict_email_phishing():
    """Test the phishing prediction endpoint with a suspicious email."""
    payload = {
        "email_text": "URGENT! Your account will be suspended. Click here now!",
        "subject": "Account Verification Required"
    }
    response = client.post("/predict/email", json=payload)
    
    # Basic checks
    assert response.status_code == 200
    data = response.json()

    # Check response structure
    assert "prediction" in data
    assert "confidence" in data
    assert "is_phishing" in data
    assert "analysis" in data

    # Check analysis structure
    analysis = data["analysis"]
    assert "suspicious_keywords_found" in analysis
    assert "urls_found" in analysis
    assert "urgency_indicators" in analysis
    assert "financial_terms" in analysis

def test_predict_email_legit():
    """Test with a normal non-phishing email."""
    payload = {
        "email_text": "Hi team, please find the meeting notes attached.",
        "subject": "Meeting Notes"
    }
    response = client.post("/predict/email", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert "prediction" in data
    assert "confidence" in data
    assert "is_phishing" in data
    assert isinstance(data["is_phishing"], int)
