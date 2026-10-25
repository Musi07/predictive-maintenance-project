# ============================================================
# LAB 6: CLASSIFICATION - DECISION TREE
# Topic: Predictive Maintenance
# ============================================================

import pandas as pd
import numpy as np
import matplotlib

# Use non-GUI backend so plots are saved directly
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# PART 1: LOAD DATASET
# ============================================================

df = pd.read_csv("data/ai4i2020.csv")

print("=" * 70)
print("LAB 6: CLASSIFICATION - DECISION TREE")
print("=" * 70)

print("\nDataset loaded successfully!")

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Columns:")
print(df.columns.tolist())


# ============================================================
# PART 2: SELECT FEATURES AND TARGET
# ============================================================

# We do not use UDI or Product ID because they are identifiers.
#
# Type is categorical, so we convert it using one-hot encoding.

features = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

target = "Machine failure"

X = df[features].copy()
y = df[target].copy()


# One-hot encode machine Type
X = pd.get_dummies(
    X,
    columns=["Type"],
    dtype=int
)

print("\nFeatures selected:")
print(X.columns.tolist())

print("\nTarget:")
print(target)

print("\nFeature matrix shape:")
print(X.shape)


# ============================================================
# PART 3: TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 70)
print("PART 3: TRAIN-TEST SPLIT")
print("=" * 70)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTraining failure distribution:")
print(y_train.value_counts())

print("\nTesting failure distribution:")
print(y_test.value_counts())


# ============================================================
# PART 4: FEATURE STANDARDIZATION
# ============================================================

# Standardization is not strictly required for Decision Trees,
# but it is included to demonstrate preprocessing for
# classification as part of the lab.

scaler = StandardScaler()

# Fit only on training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform testing data
X_test_scaled = scaler.transform(X_test)

print("\n" + "=" * 70)
print("PART 4: FEATURE STANDARDIZATION")
print("=" * 70)

print("\nTraining data standardized successfully.")
print("Testing data standardized successfully.")


# ============================================================
# PART 5: DECISION TREE USING GINI INDEX
# ============================================================

print("\n" + "=" * 70)
print("PART 5: DECISION TREE - GINI INDEX")
print("=" * 70)

gini_model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    random_state=42
)

gini_model.fit(
    X_train_scaled,
    y_train
)

gini_predictions = gini_model.predict(
    X_test_scaled
)

gini_accuracy = accuracy_score(
    y_test,
    gini_predictions
)

print("\nGini Decision Tree Accuracy:")
print(f"{gini_accuracy:.4f}")

print(f"\nGini Accuracy Percentage: {gini_accuracy * 100:.2f}%")


# ============================================================
# PART 6: GINI CONFUSION MATRIX
# ============================================================

gini_cm = confusion_matrix(
    y_test,
    gini_predictions
)

print("\nGini Confusion Matrix:")
print(gini_cm)


# ============================================================
# PART 7: GINI CLASSIFICATION REPORT
# ============================================================

print("\nGini Classification Report:")

print(
    classification_report(
        y_test,
        gini_predictions,
        target_names=[
            "No Failure",
            "Failure"
        ]
    )
)


# ============================================================
# PART 8: DECISION TREE USING ENTROPY
# ============================================================

print("\n" + "=" * 70)
print("PART 8: DECISION TREE - ENTROPY")
print("=" * 70)

entropy_model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=5,
    random_state=42
)

entropy_model.fit(
    X_train_scaled,
    y_train
)

entropy_predictions = entropy_model.predict(
    X_test_scaled
)

entropy_accuracy = accuracy_score(
    y_test,
    entropy_predictions
)

print("\nEntropy Decision Tree Accuracy:")
print(f"{entropy_accuracy:.4f}")

print(
    f"\nEntropy Accuracy Percentage: "
    f"{entropy_accuracy * 100:.2f}%"
)


# ============================================================
# PART 9: ENTROPY CONFUSION MATRIX
# ============================================================

entropy_cm = confusion_matrix(
    y_test,
    entropy_predictions
)

print("\nEntropy Confusion Matrix:")
print(entropy_cm)


# ============================================================
# PART 10: ENTROPY CLASSIFICATION REPORT
# ============================================================

print("\nEntropy Classification Report:")

print(
    classification_report(
        y_test,
        entropy_predictions,
        target_names=[
            "No Failure",
            "Failure"
        ]
    )
)


# ============================================================
# PART 11: COMPARE GINI AND ENTROPY
# ============================================================

print("\n" + "=" * 70)
print("PART 11: GINI VS ENTROPY COMPARISON")
print("=" * 70)

comparison = pd.DataFrame({
    "Criterion": [
        "Gini",
        "Entropy"
    ],
    "Accuracy": [
        gini_accuracy,
        entropy_accuracy
    ],
    "Accuracy (%)": [
        gini_accuracy * 100,
        entropy_accuracy * 100
    ]
})

print("\n")
print(comparison.to_string(index=False))


if gini_accuracy > entropy_accuracy:

    print("\nBetter performing criterion: Gini")

elif entropy_accuracy > gini_accuracy:

    print("\nBetter performing criterion: Entropy")

else:

    print("\nBoth criteria produced the same accuracy.")


# ============================================================
# PART 12: GINI DECISION TREE VISUALIZATION
# ============================================================

plt.figure(figsize=(22, 12))

plot_tree(
    gini_model,
    feature_names=X.columns,
    class_names=[
        "No Failure",
        "Failure"
    ],
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title(
    "Decision Tree - Gini Index"
)

plt.tight_layout()

plt.savefig(
    "plots/decision_tree_gini.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "\nGini decision tree visualization saved successfully."
)


# ============================================================
# PART 13: ENTROPY DECISION TREE VISUALIZATION
# ============================================================

plt.figure(figsize=(22, 12))

plot_tree(
    entropy_model,
    feature_names=X.columns,
    class_names=[
        "No Failure",
        "Failure"
    ],
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title(
    "Decision Tree - Entropy"
)

plt.tight_layout()

plt.savefig(
    "plots/decision_tree_entropy.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Entropy decision tree visualization saved successfully."
)


# ============================================================
# PART 14: FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": gini_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n" + "=" * 70)
print("PART 14: FEATURE IMPORTANCE")
print("=" * 70)

print("\nFeature Importance:")
print(
    feature_importance.to_string(index=False)
)


# ============================================================
# PART 15: FEATURE IMPORTANCE VISUALIZATION
# ============================================================

plt.figure(figsize=(10, 6))

plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")

plt.title(
    "Decision Tree Feature Importance"
)

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    "plots/decision_tree_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "\nFeature importance chart saved successfully."
)


# ============================================================
# PART 16: SAVE MODEL RESULTS
# ============================================================

results = pd.DataFrame({
    "Model": [
        "Decision Tree - Gini",
        "Decision Tree - Entropy"
    ],
    "Accuracy": [
        gini_accuracy,
        entropy_accuracy
    ],
    "Accuracy (%)": [
        gini_accuracy * 100,
        entropy_accuracy * 100
    ]
})

results.to_csv(
    "data/decision_tree_results.csv",
    index=False
)

print(
    "\nDecision Tree results saved to:"
    " data/decision_tree_results.csv"
)


# ============================================================
# PART 17: SAVE FEATURE IMPORTANCE
# ============================================================

feature_importance.to_csv(
    "data/decision_tree_feature_importance.csv",
    index=False
)

print(
    "Feature importance saved to:"
    " data/decision_tree_feature_importance.csv"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("LAB 6 SUMMARY")
print("=" * 70)

print(
    f"\nGini Accuracy: "
    f"{gini_accuracy * 100:.2f}%"
)

print(
    f"Entropy Accuracy: "
    f"{entropy_accuracy * 100:.2f}%"
)

print(
    f"\nTraining Samples: {len(X_train)}"
)

print(
    f"Testing Samples: {len(X_test)}"
)

print(
    "\nTarget variable: Machine failure"
)


# ============================================================
# COMPLETION MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("LAB 6 - DECISION TREE COMPLETED SUCCESSFULLY!")
print("=" * 70)