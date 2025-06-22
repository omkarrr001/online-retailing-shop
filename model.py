# model.py (or train.py)

import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv(r"C:\\Users\\narut\\Downloads\\online_retail_500 (1).csv")

# Drop rows with missing CustomerID
df = df.dropna(subset=["CustomerID"])

# Group by CustomerID
grouped = df.groupby("CustomerID").agg({
    "Quantity": "mean",
    "UnitPrice": "mean"
}).reset_index()

# Features for clustering
X = grouped[["Quantity", "UnitPrice"]]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X_scaled)

# Save model and scaler
joblib.dump(kmeans, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model and scaler saved successfully.")