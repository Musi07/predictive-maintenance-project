import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Dynamically resolve path relative to script directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "ai4i2020_processed.csv")

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Could not find processed dataset at: {data_path}")

df = pd.read_csv(data_path)

# Define Features and Target
drop_cols = ['UDI', 'Product ID', 'Type', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
X = df.drop(columns=[c for c in drop_cols if c in df.columns])
y = df['Machine failure']

# Train-Test Split (Stratified to handle class imbalance)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Train Gaussian Naive Bayes Model
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

# 3. Evaluate Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)

print("=== Lab 7: Gaussian Naive Bayes Results ===")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print("\nConfusion Matrix:")
print(cm)

# Save Metrics to CSV in the lab7 folder
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lab7_results.csv")
results_df = pd.DataFrame([{
    'Model': 'Gaussian Naive Bayes',
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1_Score': f1
}])
results_df.to_csv(output_path, index=False)
print(f"\nResults successfully saved to: {output_path}")