# exploratory_data_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define file paths
processed_train_data_path = 'data/processed/train_data.csv'
figures_path = 'figures'

# Create directories if they don't exist
os.makedirs(figures_path, exist_ok=True)

# Load processed training data
print("Loading processed data...")
train_data = pd.read_csv(processed_train_data_path)

# Display the first few rows of the dataset
print("Train Data:")
print(train_data.head())

# Basic statistics
print("Basic Statistics:")
print(train_data.describe())

# Fraud distribution
print("Fraud Distribution:")
print(train_data['is_fraud'].value_counts(normalize=True))

# Plot fraud distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='is_fraud', data=train_data)
plt.title('Fraud Distribution')
plt.xlabel('Fraud')
plt.ylabel('Count')
plt.savefig(os.path.join(figures_path, 'fraud_distribution.png'))
plt.show()

# Correlation matrix
print("Correlation Matrix:")
correlation_matrix = train_data.corr()
print(correlation_matrix)

# Plot correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig(os.path.join(figures_path, 'correlation_matrix.png'))
plt.show()

# Distribution of numerical features
numerical_features = ['transaction_amount', 'transaction_amount_per_day']
for feature in numerical_features:
    plt.figure(figsize=(6, 4))
    sns.histplot(train_data[feature], kde=True)
    plt.title(f'{feature.capitalize()} Distribution')
    plt.xlabel(feature.capitalize())
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(figures_path, f'{feature}_distribution.png'))
    plt.show()

# Box plots for numerical features
for feature in numerical_features:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='is_fraud', y=feature, data=train_data)
    plt.title(f'{feature.capitalize()} vs Fraud')
    plt.xlabel('Fraud')
    plt.ylabel(feature.capitalize())
    plt.savefig(os.path.join(figures_path, f'{feature}_boxplot.png'))
    plt.show()

# Categorical feature distribution
categorical_features = train_data.select_dtypes(include=['int64', 'object']).columns.drop(['is_fraud'])
for feature in categorical_features:
    plt.figure(figsize=(12, 6))
    sns.countplot(x=feature, hue='is_fraud', data=train_data)
    plt.title(f'{feature.capitalize()} vs Fraud')
    plt.xlabel(feature.capitalize())
    plt.ylabel('Count')
    plt.legend(title='Fraud', loc='upper right')
    plt.savefig(os.path.join(figures_path, f'{feature}_fraud.png'))
    plt.show()

print("Exploratory Data Analysis completed!")
