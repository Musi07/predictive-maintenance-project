import os
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

# 1. Dynamic path resolution relative to script location
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "ai4i2020_processed.csv")

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Could not find processed dataset at: {data_path}")

print(f"Loading dataset from: {data_path}")
df = pd.read_csv(data_path)

# Select operational features for clustering
features = [
    'Air temperature [K]', 
    'Process temperature [K]', 
    'Rotational speed [rpm]', 
    'Torque [Nm]', 
    'Tool wear [min]'
]
X = df[[c for c in features if c in df.columns]]

# Scale features so DBSCAN distance calculations (eps) work uniformly across all variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Optimized DBSCAN Clustering (eps=1.5 on scaled features)
print("Fitting optimized DBSCAN model...")
dbscan = DBSCAN(eps=1.5, min_samples=10)
df['DBSCAN_Cluster'] = dbscan.fit_predict(X_scaled)

# 3. Agglomerative (Hierarchical) Clustering
print("Fitting Agglomerative Clustering model...")
agg = AgglomerativeClustering(n_clusters=3)
df['Agglo_Cluster'] = agg.fit_predict(X_scaled)

print("\n=== Lab 10: Density-Based & Hierarchical Clustering Completed ===")
print("DBSCAN Cluster Counts (-1 represents noise/outliers):")
print(df['DBSCAN_Cluster'].value_counts())

print("\nAgglomerative Cluster Counts:")
print(df['Agglo_Cluster'].value_counts())