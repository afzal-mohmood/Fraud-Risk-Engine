#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np

df = pd.read_csv('fraud_dataset.csv')

print(f"Dataset Shape: {df.shape}")

df.head()

df.info()

df_cleaned = df.drop(columns=['Transaction_ID', 'User_ID'])

# Convert text Timestamps into actual Date-Time objects using 'mixed' format
# This allows Pandas to automatically figure out the format of your strings!
df_cleaned['Timestamp'] = pd.to_datetime(df_cleaned['Timestamp'], format='mixed')

# Extract the 'Hour of Day' (0 to 23) as a new numerical feature
df_cleaned['Hour_of_Day'] = df_cleaned['Timestamp'].dt.hour

# Now drop the original raw Timestamp column since we have extracted its value
df_cleaned = df_cleaned.drop(columns=['Timestamp'])

# Verify the updates by checking the new columns list
df_cleaned.head(2)

df_cleaned.info()

# Count exactly how many 0s and 1s exist in our target column
fraud_counts = df_cleaned['Fraud_Label'].value_counts()
print("--- Raw Transaction Counts ---")
print(fraud_counts)

# Calculate the percentages to see the exact lopsided ratio
fraud_percentages = df_cleaned['Fraud_Label'].value_counts(normalize=True) * 100
print("\n--- Percentage Breakdown ---")
print(fraud_percentages)

df_cleaned.nunique()

# Identify all text/categorical columns that need conversion
categorical_cols = ['Transaction_Type', 'Device_Type', 'Location', 
                    'Merchant_Category', 'Card_Type', 'Authentication_Method']

# Convert these text columns into mathematical binary flags (0 or 1)
# 'drop_first=True' is a clever statistical trick that removes redundant data columns
df_encoded = pd.get_dummies(df_cleaned, columns=categorical_cols, drop_first=True, dtype=int)

# Look at the new dimensions of our expanded dataset
print(f"Old Cleaned Layout Shape: {df_cleaned.shape}")
print(f"New Mathematically Encoded Layout Shape: {df_encoded.shape}")

# Preview the first 2 rows to see the newly generated columns
df_encoded.head(2)

df_encoded.info()

# ==========================================
# TRAIN-TEST SPLIT & MODEL TRAINING
# ==========================================
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. Separate features (X) from the true outcome column (y)
X = df_encoded.drop(columns=['Fraud_Label'])
y = df_encoded['Fraud_Label']

# 2. Split data: 80% to train our model, 20% to test its predictive ability
# 'stratify=y' ensures both the train and test sets get the exact same 70/30 fraud split!
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Initialize the Machine Learning Classifier
# We set 100 trees (n_estimators) and force it to balance the lopsided fraud classes
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

# 4. Run the training process (The model studies the data)
print("Training the Random Forest model... Please wait a few seconds...")
model.fit(X_train, y_train)
print("Model training completed successfully!")

# 5. Test the model by making predictions on the hidden 20% test data
predictions = model.predict(X_test)

# 6. Display the final score report card
print("\n--- Model Performance Evaluation Matrix ---")
print(classification_report(y_test, predictions))

# ==========================================
# DROPPING RISK_SCORE & RETRAINING
# ==========================================

# 1. Drop the cheating 'Risk_Score' column from our features matrix
X_realistic = X.drop(columns=['Risk_Score'])

# 2. Split the data again using our realistic features
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_realistic, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Retrain the model on real behavioral patterns
model_realistic = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model_realistic.fit(X_train_r, y_train_r)

# 4. Generate the new realistic predictions
predictions_realistic = model_realistic.predict(X_test_r)

# 5. Display the true report card
print("--- Realistic Model Performance (Without Cheating Column) ---")
print(classification_report(y_test_r, predictions_realistic))

# ==========================================
# EXPORTING CLEAN DATA FOR STAKEHOLDERS
# ==========================================

# Save our cleaned data frame out to a new CSV file on your hard drive
df_cleaned.to_csv('cleaned_fraud_visualization.csv', index=False)
print("Success! 'cleaned_fraud_visualization.csv' has been saved to your project folder.")





