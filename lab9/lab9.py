import os
import pandas as pd

# Set non-interactive backend BEFORE importing pyplot to avoid Tkinter errors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

# 1. Dynamic path resolution relative to this script
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "ai4i2020_processed.csv")

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Could not find processed dataset at: {data_path}")

print(f"Loading dataset from: {data_path}")
df = pd.read_csv(data_path)

# Select operational telemetry features for clustering
features = [
    'Air temperature [K]', 
    'Process temperature [K]', 
    'Rotational speed [rpm]', 
    'Torque [Nm]', 
    'Tool wear [min]'
]
X = df[[c for c in features if c in df.columns]]

# 2. Elbow Method Implementation (Testing K from 1 to 10)
print("Calculating Within-Cluster Sum of Squares (WCSS) for K=1 to 10...")
wcss = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Ensure plot save directory exists
plots_dir = os.path.join(BASE_DIR, "plots")
os.makedirs(plots_dir, exist_ok=True)
plot_path = os.path.join(plots_dir, "kmeans_elbow_plot.png")

# Generate and save the Elbow Curve plot
plt.figure(figsize=(8, 5))
plt.plot(K_range, wcss, marker='o', linestyle='--', color='b')
plt.title("Lab 9: Elbow Method for Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS (Inertia)")
plt.grid(True)
plt.savefig(plot_path)
plt.close()

# 3. Fit Final K-Means Model using Optimal Clusters (K = 3)
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['KMeans_Cluster'] = kmeans.fit_predict(X)

print("\n=== Lab 9: K-Means Clustering Completed ===")
print("Cluster Distribution:")
print(df['KMeans_Cluster'].value_counts())
print(f"\nElbow plot successfully saved to: {plot_path}")