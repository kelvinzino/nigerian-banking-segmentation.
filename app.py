import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import altair as alt  # import at top

st.title("Real Nigerian Retail Banking Customer Segmentation App")

uploaded_file = st.file_uploader("Upload Banking CSV File", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    df.drop(columns=["merchant_category_code", "merchant_name", "device_id"], inplace=True)

    customer_df = df.groupby("customer_id").agg({
        "amount_ngn": ["sum", "mean", "std", "count"],
    }).reset_index()

    # rename columns
    customer_df.columns = ["customer_id", "total_amount", "avg_amount", "std_amount", "transaction_count"]

    # Replace NaNs from std_amount with 0
    customer_df.fillna(0, inplace=True)

    # Scale data
    scaler = StandardScaler()
    x = scaler.fit_transform(customer_df[["total_amount", "avg_amount", "std_amount", "transaction_count"]])

    # Train KMeans model
    kmeans = KMeans(n_clusters=3, random_state=42)
    customer_df["cluster"] = kmeans.fit_predict(x)

    # Cluster names
    cluster_names = {
        0: "Regular Retail Users",
        1: "High Value Premium Customers",
        2: "Frequent Business Users / Merchants"
    }

    customer_df["Customer_Segment"] = customer_df["cluster"].map(cluster_names)

    # Show tables
    st.subheader("Clustered Customer Data")
    st.write(customer_df)

    st.subheader("Number of Customers In Each Segment")
    st.write(customer_df["Customer_Segment"].value_counts())

    # Scatter plot
    st.subheader("Customer Segmentation Scatter Plot")
    chart = alt.Chart(customer_df).mark_circle(size=60).encode(
        x='total_amount',
        y='transaction_count',
        color='Customer_Segment',
        tooltip=['customer_id', 'total_amount', 'transaction_count', 'Customer_Segment']
    ).interactive()

    st.altair_chart(chart, use_container_width=True)
