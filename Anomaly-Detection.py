# anomaly_detection.py

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os

# Define file paths
processed_train_data_path = 'data/processed/train_data.csv'
processed_test_data_path = 'data/processed/test_data.csv'
model_path = 'models/isolation_forest_model.pkl'
results_path = 'results/anomaly_detection_report.txt'

# Create directories if they don't exist
os.makedirs(os.path.dirname(model_path), exist_ok=True)
os.makedirs(os.path.dirname(results_path), exist_ok=True)

# Load processed training and testing data
print("Loading processed data...")
train_data = pd.read_csv(processed_train_data_path)
test_data = pd.read_csv(processed_test_data_path)

# Prepare data
X_train = train_data.drop(columns=['is_fraud'])
y_train = train_data['is_fraud']

X_test = test_data.drop(columns=['is_fraud'])
y_test = test_data['is_fraud']

# Build Isolation Forest model
print("Building Isolation Forest model...")
model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
model.fit(X_train)

# Predict anomalies on the test set
print("Predicting anomalies...")
y_pred = model.predict(X_test)
y_pred = np.where(y_pred == -1, 1, 0)  # Convert -1 (anomaly) to 1 (fraud), 1 (normal) to 0 (not fraud)

# Evaluate the model
print("Evaluating the model...")
roc_auc = roc_auc_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"ROC-AUC: {roc_auc:.4f}")
print("Classification Report:")
print(report)

# Save the evaluation report
with open(results_path, 'w') as f:
    f.write(f"ROC-AUC: {roc_auc:.4f}\n")
    f.write("Classification Report:\n")
    f.write(report)

# Save the model
print("Saving the model...")
joblib.dump(model, model_path)

print("Anomaly detection model training and evaluation completed!")
