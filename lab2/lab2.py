# ============================================================
# LAB 2: DATA PREPROCESSING AND EXPLORATION
# Topic: Predictive Maintenance
# ============================================================

import pandas as pd
import numpy as np

# ------------------------------------------------------------
# Part 1: Load the Dataset
# ------------------------------------------------------------

# Load the Predictive Maintenance dataset
df = pd.read_csv("data/ai4i2020.csv")

print("Dataset loaded successfully!")
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

# ------------------------------------------------------------
# Part 2: Check and Handle Missing Values
# ------------------------------------------------------------

print("\nMissing Values in Each Column:")
print(df.isnull().sum())

# Check whether the dataset contains any missing values
total_missing = df.isnull().sum().sum()

print("\nTotal Missing Values:", total_missing)

if total_missing == 0:
    print("There are no missing values in the dataset.")
else:
    print("Missing values were found.")

    # Remove rows containing missing values
    df = df.dropna()

    print("Rows after removing missing values:", len(df))

    # ------------------------------------------------------------
# Part 3: Check and Remove Duplicate Rows
# ------------------------------------------------------------

# Count duplicate rows
duplicate_count = df.duplicated().sum()

print("\nNumber of Duplicate Rows:", duplicate_count)

# Remove duplicate rows if they exist
if duplicate_count > 0:
    df = df.drop_duplicates()
    print("Duplicate rows removed.")
else:
    print("There are no duplicate rows in the dataset.")

print("Dataset shape after duplicate handling:", df.shape)

# ------------------------------------------------------------
# Part 4: Encode Categorical Variables
# ------------------------------------------------------------

print("\nCategorical Columns:")
print(df.select_dtypes(include=["object"]).columns.tolist())

# Create a copy so that the original dataframe remains unchanged
encoded_df = df.copy()

# One-hot encode the Type column
encoded_df = pd.get_dummies(
    encoded_df,
    columns=["Type"],
    dtype=int
)

print("\nDataset after One-Hot Encoding:")
print(encoded_df.head())

print("\nNew Dataset Shape:")
print(encoded_df.shape)

print("\nNew Column Names:")
print(encoded_df.columns.tolist())

# ------------------------------------------------------------
# Part 5: Feature Scaling - Normalization
# ------------------------------------------------------------

from sklearn.preprocessing import MinMaxScaler

# Select numerical machine features for scaling
numerical_features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

# Create a copy of the encoded dataset
normalized_df = encoded_df.copy()

# Apply Min-Max Normalization
scaler = MinMaxScaler()

normalized_df[numerical_features] = scaler.fit_transform(
    normalized_df[numerical_features]
)

print("\nNormalized Features:")
print(normalized_df[numerical_features].head())

print("\nMinimum values after normalization:")
print(normalized_df[numerical_features].min())

print("\nMaximum values after normalization:")
print(normalized_df[numerical_features].max())

# ------------------------------------------------------------
# Part 6: Feature Scaling - Standardization
# ------------------------------------------------------------

from sklearn.preprocessing import StandardScaler

# Create a copy of the encoded dataset
standardized_df = encoded_df.copy()

# Create StandardScaler
standard_scaler = StandardScaler()

# Apply standardization
standardized_df[numerical_features] = standard_scaler.fit_transform(
    standardized_df[numerical_features]
)

print("\nStandardized Features:")
print(standardized_df[numerical_features].head())

print("\nMean after standardization:")
print(standardized_df[numerical_features].mean())

print("\nStandard deviation after standardization:")
print(standardized_df[numerical_features].std())

# ------------------------------------------------------------
# Part 7: Descriptive Statistics
# ------------------------------------------------------------

print("\nDescriptive Statistics:")
print(df.describe())


# ------------------------------------------------------------
# Part 8: Save the Processed Dataset
# ------------------------------------------------------------

# Save the encoded dataset
encoded_df.to_csv(
    "data/ai4i2020_processed.csv",
    index=False
)

print("\nProcessed dataset successfully saved!")
print("File: data/ai4i2020_processed.csv")