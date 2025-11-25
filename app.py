import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import altair as alt

st.title("Real Nigerian Retail Banking Customer Segmentation App")

# Load processed CSV directly from file path
file_path = file_path = "processed_customers.csv"
customer_df = pd.read_csv(file_path)

st.subheader("Original Processed Customer Data")
st.write(customer_df.head())

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
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import altair as alt

st.title("Real Nigerian Retail Banking Customer Segmentation App")

# Load processed CSV directly from file path
file_path = r"C:\Users\LENOVO\Documents\python_class\customer\processed_customers.csv"
customer_df = pd.read_csv(file_path)

st.subheader("Original Processed Customer Data")
st.write(customer_df.head())

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
