# random_forest.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import joblib
import os

# Define file paths
processed_train_data_path = 'data/processed/train_data.csv'
processed_test_data_path = 'data/processed/test_data.csv'
model_path = 'models/random_forest_model.pkl'
results_path = 'results/random_forest_report.txt'

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

# Build Random Forest model
print("Building Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
print("Predicting on the test set...")
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluate the model
print("Evaluating the model...")
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print("Classification Report:")
print(report)

# Save the evaluation report
with open(results_path, 'w') as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")
    f.write(f"ROC-AUC: {roc_auc:.4f}\n")
    f.write("Classification Report:\n")
    f.write(report)

# Save the model
print("Saving the model...")
joblib.dump(model, model_path)

print("Random forest model training and evaluation completed!")
