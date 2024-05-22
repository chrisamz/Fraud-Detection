# data_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os

# Define file paths
raw_data_path = 'data/raw/transactions.csv'
processed_train_data_path = 'data/processed/train_data.csv'
processed_test_data_path = 'data/processed/test_data.csv'

# Create directories if they don't exist
os.makedirs(os.path.dirname(processed_train_data_path), exist_ok=True)

# Load raw data
print("Loading raw data...")
data = pd.read_csv(raw_data_path)

# Display the first few rows of the dataset
print("Raw Data:")
print(data.head())

# Data Cleaning
print("Cleaning data...")

# Handle missing values
data.fillna(data.mean(), inplace=True)

# Encode categorical features
print("Encoding categorical features...")
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# Feature Engineering
print("Performing feature engineering...")

# Example: Create new features based on existing data
data['transaction_amount_per_day'] = data['transaction_amount'] / (data['transaction_time'] / (24 * 60 * 60))

# Normalize numerical features
print("Normalizing numerical features...")
scaler = StandardScaler()
numerical_features = ['transaction_amount', 'transaction_amount_per_day']
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Split the data into training and testing sets
print("Splitting data into training and testing sets...")
X = data.drop(columns=['transaction_id', 'is_fraud'])
y = data['is_fraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save processed data
print("Saving processed data...")
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

train_data.to_csv(processed_train_data_path, index=False)
test_data.to_csv(processed_test_data_path, index=False)

print("Data preprocessing completed!")
