import os
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "ai4i2020_processed.csv")

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Could not find processed dataset at: {data_path}")

print(f"Loading dataset from: {data_path}")
df = pd.read_csv(data_path)

drop_cols = ['UDI', 'Product ID', 'Type', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
X = df.drop(columns=[c for c in drop_cols if c in df.columns])
y = df['Machine failure']

# 1. Isolation Forest Anomaly Detection
print("Running Isolation Forest Anomaly Detection...")
iso = IsolationForest(contamination=0.03, random_state=42)
df['Anomaly_Flag'] = iso.fit_predict(X)

print("\n=== Lab 11: Outlier Detection Results ===")
print(f"Anomalies Detected (Outliers): {sum(df['Anomaly_Flag'] == -1)}")

# 2. Random Forest Feature Importance
print("\nTraining Random Forest for Feature Importance Ranking...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importances:")
print(importance_df)

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lab11_feature_importance.csv")
importance_df.to_csv(output_path, index=False)
print(f"\nFeature importances saved to: {output_path}")