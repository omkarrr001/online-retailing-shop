# model.py

import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv(r"C:\Users\saikh\Downloads\online_retail_500.csv")

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

# Rename columns for clarity
grouped.rename(columns={
    "Quantity": "AvgQuantity",
    "UnitPrice": "AvgUnitPrice",
    "TotalSpend": "TotalSpend",
    "InvoiceNo": "InvoiceCount"
}, inplace=True)

# Features for clustering
features = ["AvgQuantity", "AvgUnitPrice", "TotalSpend", "InvoiceCount"]
X = grouped[features]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train KMeans model
kmeans = KMeans(n_clusters=4, random_state=42)
grouped["Cluster"] = kmeans.fit_predict(X_scaled)

# Save model and scaler
joblib.dump(kmeans, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Save clustered data to CSV
grouped.to_csv("clustered_customers.csv", index=False)

# Print summary
print("✅ Model trained successfully!")
print(f"📊 Features used: {features}")
print("📁 model.pkl and scaler.pkl saved.")
print("📄 Clustered customer data saved to clustered_customers.csv")
print("\n🔢 Cluster counts:")
print(grouped["Cluster"].value_counts())
