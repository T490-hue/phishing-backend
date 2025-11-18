import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer  # <-- ADD THIS
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from ucimlrepo import fetch_ucirepo
import joblib
import re
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print(" HYBRID PHISHING DETECTION MODEL - EMAIL + WEBSITE FEATURES")
print("=" * 80)
print(" Dataset 1: UCI Phishing Websites (Website Features)")
print(" Dataset 2: Phishing Email Dataset (Email Text Features)")
print("=" * 80)

# ============================================================================
# PART 1: LOAD UCI PHISHING WEBSITES DATASET
# ============================================================================
print("\n[1/7] Loading UCI Phishing Websites Dataset...")
try:
    phishing_websites = fetch_ucirepo(id=327)
    X_uci = phishing_websites.data.features
    y_uci = phishing_websites.data.targets.squeeze().replace(-1, 0)
    X_uci.columns = X_uci.columns.str.lower()
    
    print(f"✓ UCI Dataset loaded successfully")
    print(f"  → Shape: {X_uci.shape}")
    print(f"  → Phishing: {sum(y_uci==1)}, Legitimate: {sum(y_uci==0)}")
except Exception as e:
    print(f"✗ Failed to load UCI data: {e}")
    exit(1)

# ============================================================================
# PART 2: LOAD PHISHING EMAIL DATASET
# ============================================================================
print("\n[2/7] Loading Phishing Email Dataset...")
try:
    email_data = pd.read_csv('data/phishing_email.csv')
    print(f"✓ Email dataset loaded successfully")
    print(f"  → Shape: {email_data.shape}")
    print(f"  → Columns: {list(email_data.columns)}")
    print(f"  First few rows:")
    print(email_data.head(2))
    
except FileNotFoundError:
    print("✗ Email dataset not found at data/phishing_email.csv")
    print("\n  Trying alternative filenames...")
    
    possible_names = [
        'data/Phishing_Email.csv',
        'data/phishing_emails.csv',
        'data/email_phishing.csv',
        'data/spam_ham.csv'
    ]
    
    email_data = None
    for filename in possible_names:
        try:
            email_data = pd.read_csv(filename)
            print(f"✓ Found dataset: {filename}")
            break
        except:
            continue
    
    if email_data is None:
        print("✗ Could not find email dataset. Please check the filename.")
        print("  Available files in data/:")
        import os
        print(os.listdir('data/'))
        exit(1)

# ============================================================================
# PART 3: EMAIL TEXT FEATURE EXTRACTION
# ============================================================================
print("\n[3/7] Extracting Features from Email Text...")

def extract_email_features(text):
    """Extract numerical features from email text"""
    if pd.isna(text) or text == '':
        text = ''
    
    text = str(text).lower()
    
    features = {}
    
    # 1. Length-based features
    features['email_length'] = len(text)
    features['word_count'] = len(text.split())
    features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
    
    # 2. Suspicious keywords (phishing indicators)
    phishing_keywords = [
        'urgent', 'verify', 'account', 'suspended', 'click', 'confirm',
        'password', 'credit', 'bank', 'security', 'update', 'expire',
        'winner', 'prize', 'congratulations', 'claim', 'free', 'offer'
    ]
    features['suspicious_keywords'] = sum(keyword in text for keyword in phishing_keywords)
    
    # 3. URL presence
    features['has_url'] = 1 if re.search(r'http[s]?://', text) else 0
    features['url_count'] = len(re.findall(r'http[s]?://', text))
    
    # 4. Special characters (common in phishing)
    features['exclamation_count'] = text.count('!')
    features['question_count'] = text.count('?')
    features['dollar_sign'] = text.count('$')
    features['at_symbol'] = text.count('@')
    
    # 5. Uppercase ratio (phishing emails often use ALL CAPS)
    if text:
        features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text)
    else:
        features['uppercase_ratio'] = 0
    
    # 6. Number presence
    features['has_numbers'] = 1 if re.search(r'\d', text) else 0
    features['number_count'] = len(re.findall(r'\d+', text))
    
    # 7. Email-like patterns
    features['email_addresses'] = len(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text))
    
    # 8. Sense of urgency words
    urgent_words = ['urgent', 'immediate', 'action required', 'act now', 'limited time']
    features['urgency_score'] = sum(word in text for word in urgent_words)
    
    # 9. Financial terms
    financial_terms = ['money', 'payment', 'transfer', 'account', 'bank', 'credit card']
    features['financial_terms'] = sum(term in text for term in financial_terms)
    
    # 10. Generic greeting (phishing often lacks personalization)
    generic_greetings = ['dear customer', 'dear user', 'dear member', 'valued customer']
    features['generic_greeting'] = 1 if any(greeting in text for greeting in generic_greetings) else 0
    
    return features

# Identify text columns in email dataset
text_columns = []
for col in email_data.columns:
    if email_data[col].dtype == 'object':
        text_columns.append(col)

print(f"  → Found text columns: {text_columns}")

# Find text field
if 'Email Text' in email_data.columns:
    text_field = 'Email Text'
elif 'email_text' in email_data.columns:
    text_field = 'email_text'
elif 'body' in email_data.columns:
    text_field = 'body'
elif 'text' in email_data.columns:
    text_field = 'text'
elif 'text_combined' in email_data.columns:
    text_field = 'text_combined'
else:
    text_field = text_columns[0] if text_columns else email_data.columns[0]

print(f"  → Using text field: '{text_field}'")

# Extract features
print("  → Extracting numerical features from email text...")
email_features_list = []
for idx, text in enumerate(email_data[text_field]):
    if idx % 10000 == 0:
        print(f"    Processing {idx}/{len(email_data)}...")
    email_features_list.append(extract_email_features(text))

X_email = pd.DataFrame(email_features_list)
print(f"✓ Email features extracted: {X_email.shape}")

# Get labels
label_col = None
for col in email_data.columns:
    if 'label' in col.lower() or 'spam' in col.lower() or 'phishing' in col.lower():
        label_col = col
        break

if label_col:
    y_email = email_data[label_col]
    if y_email.dtype == 'object':
        label_mapping = {
            'spam': 1, 'phishing': 1, 'ham': 0, 'legitimate': 0,
            'Spam': 1, 'Phishing': 1, 'Ham': 0, 'Legitimate': 0,
            1: 1, 0: 0, '1': 1, '0': 0
        }
        y_email = y_email.map(label_mapping)
    y_email = y_email.astype(int)
else:
    print("✗ Could not find label column")
    exit(1)

print(f"  → Email labels: Phishing={sum(y_email==1)}, Legitimate={sum(y_email==0)}")

# ============================================================================
# PART 4: BALANCE AND SAMPLE DATASETS
# ============================================================================
print("\n[4/7] Balancing and Sampling Datasets...")

sample_size = min(len(X_uci), len(X_email))
email_indices = np.random.choice(len(X_email), size=sample_size, replace=False)
X_email_sampled = X_email.iloc[email_indices].reset_index(drop=True)
y_email_sampled = y_email.iloc[email_indices].reset_index(drop=True)

print(f"✓ Sampled email dataset: {X_email_sampled.shape}")

# Normalize email features
scaler = StandardScaler()
X_email_normalized = pd.DataFrame(
    scaler.fit_transform(X_email_sampled),
    columns=X_email_sampled.columns
)

# Convert to -1, 0, 1 scale
for col in X_email_normalized.columns:
    X_email_normalized[col] = pd.cut(
        X_email_normalized[col],
        bins=[-np.inf, -0.5, 0.5, np.inf],
        labels=[-1, 0, 1]
    ).astype(int)

print(f"✓ Features normalized to UCI scale")

# ============================================================================
# PART 5: COMBINE DATASETS (WITH PROPER HANDLING)
# ============================================================================
print("\n[5/7] Creating Hybrid Dataset...")

# Add source identifier
X_uci['source_website'] = 1
X_email_normalized['source_website'] = 0

# Get all unique columns from both datasets
all_columns = list(set(X_uci.columns) | set(X_email_normalized.columns))
print(f"  → Total unique features: {len(all_columns)}")

# Fill missing columns with 0 for both datasets
for col in all_columns:
    if col not in X_uci.columns:
        X_uci[col] = 0
    if col not in X_email_normalized.columns:
        X_email_normalized[col] = 0

# Ensure same column order
X_uci = X_uci[all_columns]
X_email_normalized = X_email_normalized[all_columns]

# NOW combine
X_combined = pd.concat([X_uci, X_email_normalized], axis=0, ignore_index=True)
y_combined = pd.concat([y_uci, y_email_sampled], axis=0, ignore_index=True)

print(f"✓ Hybrid dataset created")
print(f"  → Total samples: {X_combined.shape[0]}")
print(f"  → Total features: {X_combined.shape[1]}")
print(f"  → Phishing: {sum(y_combined==1)}, Legitimate: {sum(y_combined==0)}")
print(f"  → From websites: {sum(X_combined['source_website']==1)}")
print(f"  → From emails: {sum(X_combined['source_website']==0)}")

# Check for NaN values
print(f"\n  → Checking for missing values...")
nan_count = X_combined.isna().sum().sum()
if nan_count > 0:
    print(f"    WARNING: Found {nan_count} NaN values")
    print(f"    Filling with 0...")
    X_combined = X_combined.fillna(0)
    print(f"    ✓ All NaN values filled")
else:
    print(f"    ✓ No missing values found")

# Save feature columns
model_columns = list(X_combined.columns)
joblib.dump(model_columns, 'model_columns.joblib')
joblib.dump(scaler, 'email_scaler.joblib')
print(f"✓ Saved feature columns and scaler")

# ============================================================================
# PART 6: TRAIN MULTIPLE MODELS
# ============================================================================
print("\n[6/7] Training Models on Hybrid Dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y_combined, test_size=0.2, random_state=42, stratify=y_combined
)

# Double-check for NaN in training data
if X_train.isna().any().any():
    print("  → Filling remaining NaN in training data...")
    X_train = X_train.fillna(0)
if X_test.isna().any().any():
    print("  → Filling remaining NaN in test data...")
    X_test = X_test.fillna(0)

mlflow.set_experiment("Phishing-Detection-Hybrid-Email-Website")

models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

best_model = None
best_accuracy = 0
best_model_name = ''

with mlflow.start_run() as run:
    print(f"\n  MLflow Run ID: {run.info.run_id}")
    
    for model_name, model in models.items():
        print(f"\n  Training {model_name}...")
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        print(f"    → Accuracy:  {acc:.4f}")
        print(f"    → Precision: {prec:.4f}")
        print(f"    → Recall:    {rec:.4f}")
        print(f"    → F1-Score:  {f1:.4f}")
        
        # Log to MLflow
        mlflow.log_param(f"{model_name}_type", model_name)
        mlflow.log_metric(f"{model_name}_accuracy", acc)
        mlflow.log_metric(f"{model_name}_precision", prec)
        mlflow.log_metric(f"{model_name}_recall", rec)
        mlflow.log_metric(f"{model_name}_f1", f1)
        
        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            best_model_name = model_name
    
    print(f"\n  🏆 Best Model: {best_model_name}")
    print(f"     Accuracy: {best_accuracy:.4f}")
    
    # Log best model
    mlflow.sklearn.log_model(best_model, "model")
    mlflow.log_param("dataset_source", "UCI Websites + Email Dataset (Hybrid)")
    mlflow.log_param("total_samples", len(X_combined))
    mlflow.log_param("n_features", len(model_columns))
    mlflow.log_param("best_model", best_model_name)
    
    # Save model
    joblib.dump(best_model, 'phishing_model.joblib')
    print(f"\n  ✓ Model saved to 'phishing_model.joblib'")

# ============================================================================
# PART 7: DETAILED EVALUATION
# ============================================================================
print("\n[7/7] Detailed Model Evaluation...")

y_pred_final = best_model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_final, target_names=['Legitimate', 'Phishing']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred_final)
print(f"  True Negatives:  {cm[0][0]}")
print(f"  False Positives: {cm[0][1]}")
print(f"  False Negatives: {cm[1][0]}")
print(f"  True Positives:  {cm[1][1]}")

print("\n" + "=" * 80)
print(" TRAINING COMPLETE!")
print("=" * 80)
print(f" ✓ Model: {best_model_name}")
print(f" ✓ Accuracy: {best_accuracy:.4f}")
print(f" ✓ Files saved:")
print(f"   - phishing_model.joblib")
print(f"   - model_columns.joblib")
print(f"   - email_scaler.joblib")
print("=" * 80)
