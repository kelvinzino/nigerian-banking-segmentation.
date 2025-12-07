import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import altair as alt

st.title("Real Nigerian Retail Banking Customer Segmentation App")

# Load processed CSV
file_path = "processed_customers.csv"
customer_df = pd.read_csv(file_path)

st.subheader("Original Processed Customer Data")
st.write(customer_df.head())

# ----------------------------
# PREPARE DATA & TRAIN MODEL
# ----------------------------
features = ["total_amount", "avg_amount", "std_amount", "transaction_count"]

scaler = StandardScaler()
x_scaled = scaler.fit_transform(customer_df[features])

kmeans = KMeans(n_clusters=3, random_state=42)
customer_df["cluster"] = kmeans.fit_predict(x_scaled)

# Cluster labels
cluster_names = {
    0: "Regular Retail Users",
    1: "High Value Premium Customers",
    2: "Frequent Business Users / Merchants"
}
customer_df["Customer_Segment"] = customer_df["cluster"].map(cluster_names)

# ----------------------------
# SHOW CLUSTERED DATA
# ----------------------------
st.subheader("Clustered Customer Data")
st.write(customer_df)

st.subheader("Number of Customers In Each Segment")
st.bar_chart(customer_df["Customer_Segment"].value_counts())

# ----------------------------
# SCATTER PLOT FOR SEGMENTATION
# ----------------------------
st.subheader("Customer Segmentation Scatter Plot")

chart = (
    alt.Chart(customer_df)
    .mark_circle(size=60)
    .encode(
        x=alt.X("total_amount", title="Total Amount Spent"),
        y=alt.Y("transaction_count", title="Transaction Count"),
        color=alt.Color("Customer_Segment", legend=alt.Legend(title="Customer Segments")),
        tooltip=["customer_id", "total_amount", "transaction_count", "Customer_Segment"]
    )
    .interactive()
)

st.altair_chart(chart, use_container_width=True)

# ----------------------------
# USER INPUT FOR AUTOMATIC CLUSTER PREDICTION
# ----------------------------
st.subheader("🔍 Predict Customer Segment From User Input")

st.write("Enter the customer's transaction summary below to see which segment they belong to.")

# User form
with st.form("prediction_form"):
    total_amount = st.number_input("Total Amount (₦)", min_value=0.0, step=100.0)
    avg_amount = st.number_input("Average Transaction Amount (₦)", min_value=0.0, step=10.0)
    std_amount = st.number_input("Transaction Amount Variance", min_value=0.0, step=10.0)
    transaction_count = st.number_input("Number of Transactions", min_value=0, step=1)

    submit_button = st.form_submit_button("Predict Segment")

if submit_button:
    # Prepare input for prediction
    user_data = np.array([[total_amount, avg_amount, std_amount, transaction_count]])
    user_scaled = scaler.transform(user_data)
    segment = kmeans.predict(user_scaled)[0]
    segment_name = cluster_names[segment]

    st.success(f"🎉 This customer belongs to: **{segment_name}**")
