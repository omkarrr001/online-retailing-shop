# app.py

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")


# Load data
df = pd.read_csv("online_retail_500.csv")
df_clean = df.dropna(subset=["CustomerID"])

# Sidebar navigation
page = st.sidebar.radio("Go to", ["Home", "Dataset", "Summary", "Predict"])

st.title("🛍️ Online Retail - Customer Segmentation")

if page == "Home":
    st.header("Welcome 👋")
    st.markdown("""
        This Streamlit app performs customer segmentation using KMeans clustering based on:
        - Average Quantity
        - Average Unit Price

        Navigate through the sidebar to explore the data and make predictions.
    """)

elif page == "Dataset":
    st.header("📦 Raw Dataset")
    st.dataframe(df.head(50))

elif page == "Summary":
    st.header("📊 Summary Statistics")
    st.write(df.describe())

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

elif page == "Predict":
    st.header("🧠 Predict Customer Cluster")

    # Form layout
    with st.form("predict_form"):
        customer_id = st.number_input("Customer ID", min_value=1.0, step=1.0)
        quantity = st.slider("Average Quantity", 0, 100, 10)
        unit_price = st.slider("Average Unit Price", 0, 100, 10)

        submitted = st.form_submit_button("Predict Cluster")

    if submitted:
        # Prepare data
        input_df = pd.DataFrame([[quantity, unit_price]], columns=["Quantity", "UnitPrice"])
        input_scaled = scaler.transform(input_df)
        cluster = model.predict(input_scaled)[0]

        st.success(f"🎯 Predicted Cluster: {cluster}")

        # Cluster plot
        grouped = df_clean.groupby("CustomerID").agg({
            "Quantity": "mean",
            "UnitPrice": "mean"
        }).reset_index()

        X_scaled = scaler.transform(grouped[["Quantity", "UnitPrice"]])
        grouped["Cluster"] = model.predict(X_scaled)

        fig, ax = plt.subplots()
        scatter = ax.scatter(grouped["Quantity"], grouped["UnitPrice"], c=grouped["Cluster"], cmap='viridis', alpha=0.6)
        ax.scatter(quantity, unit_price, c='red', s=100, edgecolors='black', label="Your Input")
        ax.set_xlabel("Avg Quantity")
        ax.set_ylabel("Avg Unit Price")
        ax.set_title("Customer Segments")
        ax.legend()
        st.pyplot(fig)
