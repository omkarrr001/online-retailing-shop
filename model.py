# model.py

import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv(r"C:\Users\narut\Downloads\online_retail_500.csv")

# Drop rows with missing CustomerID
df = df.dropna(subset=["CustomerID"])

# Create a TotalSpend column
df["TotalSpend"] = df["Quantity"] * df["UnitPrice"]

# Group by CustomerID
grouped = df.groupby("CustomerID").agg({
    "Quantity": "mean",
    "UnitPrice": "mean",
    "TotalSpend": "sum",
    "InvoiceNo": "nunique"
}).reset_index()

# Rename columns to match app.py
grouped.rename(columns={
    "Quantity": "AvgQuantity",
    "UnitPrice": "AvgUnitPrice",
    "TotalSpend": "TotalSpend",
    "InvoiceNo": "InvoiceCount"
}, inplace=True)

# Features for clustering (must match app.py input)
features = ["AvgQuantity", "AvgUnitPrice", "TotalSpend", "InvoiceCount"]
X = grouped[features]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
grouped["Cluster"] = kmeans.fit_predict(X_scaled)

# Save model and scaler
joblib.dump(kmeans, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Save clustered data
grouped.to_csv("clustered_customers.csv", index=False)

print("✅ Model trained successfully with extended features.")
print("📁 model.pkl, scaler.pkl, and clustered_customers.csv saved.")
print("🔢 Cluster counts:")
print(grouped["Cluster"].value_counts())


