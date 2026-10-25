import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Dynamic directory resolution
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "ai4i2020_processed.csv")

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Could not find processed dataset at: {data_path}")

df = pd.read_csv(data_path)
drop_cols = ['UDI', 'Product ID', 'Type', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
X = df.drop(columns=[c for c in drop_cols if c in df.columns])
y = df['Machine failure']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

kernels = ['linear', 'rbf']
results = []

for kernel in kernels:
    model = SVC(kernel=kernel, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    results.append({
        'Kernel': kernel,
        'Accuracy': accuracy_score(y_test, preds),
        'Precision': precision_score(y_test, preds, zero_division=0),
        'Recall': recall_score(y_test, preds, zero_division=0),
        'F1_Score': f1_score(y_test, preds, zero_division=0)
    })

results_df = pd.DataFrame(results)
print("=== Lab 8: Support Vector Machine (SVM) Results ===")
print(results_df)

# Output path setup
output_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(output_dir, "lab8_svm_results.csv")
results_df.to_csv(output_path, index=False)
print(f"\nResults saved to: {output_path}")