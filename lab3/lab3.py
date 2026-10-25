# ============================================================
# LAB 3: EXPLORATORY DATA ANALYSIS AND VISUALIZATION
# Topic: Predictive Maintenance
# ============================================================

import pandas as pd
import numpy as np

# Use a non-GUI backend so plots are saved directly as PNG files
# This avoids Tkinter/Tcl errors on some Python installations.
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ============================================================
# PART 1: LOAD DATASET
# ============================================================

df = pd.read_csv("data/ai4i2020.csv")

print("=" * 60)
print("LAB 3: EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print("\nDataset loaded successfully!")

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Columns:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())


# ============================================================
# PART 2: MACHINE FAILURE DISTRIBUTION
# Bar Chart
# ============================================================

failure_counts = df["Machine failure"].value_counts().sort_index()

plt.figure(figsize=(7, 5))

plt.bar(
    ["No Failure", "Failure"],
    failure_counts.values
)

plt.title("Machine Failure Distribution")
plt.xlabel("Machine Failure")
plt.ylabel("Number of Machines")

plt.tight_layout()

plt.savefig("plots/machine_failure_distribution.png")
plt.close()

print("\n" + "=" * 60)
print("PART 2: MACHINE FAILURE DISTRIBUTION")
print("=" * 60)

print("\nMachine Failure Counts:")
print(failure_counts)

print("\nMachine failure chart saved successfully.")


# ============================================================
# PART 3: AIR TEMPERATURE DISTRIBUTION
# Histogram
# ============================================================

plt.figure(figsize=(8, 5))

plt.hist(
    df["Air temperature [K]"],
    bins=30,
    edgecolor="black"
)

plt.title("Distribution of Air Temperature")
plt.xlabel("Air Temperature [K]")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig("plots/air_temperature_histogram.png")
plt.close()

print("\nAir temperature histogram saved successfully.")


# ============================================================
# PART 4: PROCESS TEMPERATURE DISTRIBUTION
# Histogram
# ============================================================

plt.figure(figsize=(8, 5))

plt.hist(
    df["Process temperature [K]"],
    bins=30,
    edgecolor="black"
)

plt.title("Distribution of Process Temperature")
plt.xlabel("Process Temperature [K]")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig("plots/process_temperature_histogram.png")
plt.close()

print("\nProcess temperature histogram saved successfully.")


# ============================================================
# PART 5: MACHINE TYPE DISTRIBUTION
# Bar Chart
# ============================================================

type_counts = df["Type"].value_counts().sort_index()

plt.figure(figsize=(7, 5))

plt.bar(
    type_counts.index,
    type_counts.values
)

plt.title("Machine Type Distribution")
plt.xlabel("Machine Type")
plt.ylabel("Number of Machines")

plt.tight_layout()

plt.savefig("plots/machine_type_distribution.png")
plt.close()

print("\nMachine Type Counts:")
print(type_counts)

print("\nMachine type chart saved successfully.")


# ============================================================
# PART 6: ROTATIONAL SPEED VS TORQUE
# Scatter Plot
# ============================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Rotational speed [rpm]"],
    df["Torque [Nm]"],
    alpha=0.5
)

plt.title("Rotational Speed vs Torque")
plt.xlabel("Rotational Speed [rpm]")
plt.ylabel("Torque [Nm]")

plt.tight_layout()

plt.savefig("plots/speed_vs_torque_scatter.png")
plt.close()

print("\nSpeed vs Torque scatter plot saved successfully.")


# ============================================================
# PART 7: TOOL WEAR DISTRIBUTION
# Box Plot
# ============================================================

plt.figure(figsize=(7, 5))

plt.boxplot(
    df["Tool wear [min]"],
    vert=True
)

plt.title("Tool Wear Distribution")
plt.ylabel("Tool Wear [min]")

plt.tight_layout()

plt.savefig("plots/tool_wear_boxplot.png")
plt.close()

print("\nTool wear box plot saved successfully.")


# ============================================================
# PART 8: TORQUE DISTRIBUTION
# Box Plot
# ============================================================

plt.figure(figsize=(7, 5))

plt.boxplot(
    df["Torque [Nm]"],
    vert=True
)

plt.title("Torque Distribution")
plt.ylabel("Torque [Nm]")

plt.tight_layout()

plt.savefig("plots/torque_boxplot.png")
plt.close()

print("\nTorque box plot saved successfully.")


# ============================================================
# PART 9: MACHINE FAILURE VS TOOL WEAR
# Box Plot
# ============================================================

failure_0 = df[df["Machine failure"] == 0]["Tool wear [min]"]
failure_1 = df[df["Machine failure"] == 1]["Tool wear [min]"]

plt.figure(figsize=(7, 5))

plt.boxplot(
    [failure_0, failure_1],
    tick_labels=["No Failure", "Failure"]
)

plt.title("Tool Wear by Machine Failure")
plt.xlabel("Machine Failure")
plt.ylabel("Tool Wear [min]")

plt.tight_layout()

plt.savefig("plots/tool_wear_vs_failure.png")
plt.close()

print("\nTool wear vs failure box plot saved successfully.")


# ============================================================
# PART 10: CORRELATION ANALYSIS
# Correlation Heatmap
# ============================================================

numerical_columns = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure"
]

correlation_matrix = df[numerical_columns].corr()

print("\n" + "=" * 60)
print("PART 10: CORRELATION MATRIX")
print("=" * 60)

print("\nCorrelation Matrix:")
print(correlation_matrix)


# Create correlation heatmap
plt.figure(figsize=(9, 7))

plt.imshow(
    correlation_matrix,
    interpolation="nearest",
    aspect="auto"
)

plt.colorbar()

plt.xticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig("plots/correlation_heatmap.png")
plt.close()

print("\nCorrelation heatmap saved successfully.")


# ============================================================
# PART 11: OUTLIER DETECTION USING IQR
# ============================================================

print("\n" + "=" * 60)
print("PART 11: OUTLIER DETECTION")
print("=" * 60)

outlier_columns = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

for column in outlier_columns:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower_bound) |
        (df[column] > upper_bound)
    ]

    print("\nFeature:", column)
    print("Q1:", Q1)
    print("Q3:", Q3)
    print("IQR:", IQR)
    print("Lower Bound:", lower_bound)
    print("Upper Bound:", upper_bound)
    print("Number of Outliers:", len(outliers))


# ============================================================
# PART 12: FAILURE RATE BY MACHINE TYPE
# ============================================================

failure_by_type = df.groupby("Type")["Machine failure"].mean()

print("\n" + "=" * 60)
print("PART 12: FAILURE RATE BY MACHINE TYPE")
print("=" * 60)

print("\nFailure Rate by Machine Type:")
print(failure_by_type)


plt.figure(figsize=(7, 5))

plt.bar(
    failure_by_type.index,
    failure_by_type.values
)

plt.title("Machine Failure Rate by Machine Type")
plt.xlabel("Machine Type")
plt.ylabel("Failure Rate")

plt.tight_layout()

plt.savefig("plots/failure_rate_by_type.png")
plt.close()

print("\nFailure rate chart saved successfully.")


# ============================================================
# PART 13: FEATURE RELATIONSHIP WITH MACHINE FAILURE
# ============================================================

failure_means = df.groupby("Machine failure")[
    [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]
].mean()

print("\n" + "=" * 60)
print("PART 13: FEATURE MEANS BY MACHINE FAILURE")
print("=" * 60)

print("\nAverage Feature Values by Machine Failure:")
print(failure_means)


# ============================================================
# PART 14: SAVE EDA SUMMARY
# ============================================================

summary = {
    "Total Records": len(df),
    "Total Features": len(df.columns),
    "Total Machine Failures": int(df["Machine failure"].sum()),
    "Total Normal Machines": int((df["Machine failure"] == 0).sum()),
    "Failure Rate": df["Machine failure"].mean(),
}

print("\n" + "=" * 60)
print("EDA SUMMARY")
print("=" * 60)

for key, value in summary.items():
    print(f"{key}: {value}")


# ============================================================
# COMPLETION MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("LAB 3 COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nAll visualization files have been saved inside:")
print("plots/")