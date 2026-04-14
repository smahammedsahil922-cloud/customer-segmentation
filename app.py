import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

st.title("🛒 Customer Segmentation System")

# Generate sample data
np.random.seed(42)

data = {
    "Customer_ID": range(1, 101),
    "Annual_Income": np.random.randint(20000, 100000, 100),
    "Spending_Score": np.random.randint(1, 100, 100)
}

df = pd.DataFrame(data)

# Show data
st.subheader("📊 Customer Data")
st.dataframe(df)

# K-Means Clustering
k = st.slider("Select Number of Segments (K)", 2, 5, 3)

kmeans = KMeans(n_clusters=k, random_state=0)
df["Cluster"] = kmeans.fit_predict(df[["Annual_Income", "Spending_Score"]])

# Show clustered data
st.subheader("🧠 Segmented Customers")
st.dataframe(df)

# Plot
st.subheader("📈 Visualization")

fig, ax = plt.subplots()

for i in range(k):
    cluster = df[df["Cluster"] == i]
    ax.scatter(cluster["Annual_Income"], cluster["Spending_Score"], label=f"Cluster {i}")

ax.set_xlabel("Income")
ax.set_ylabel("Spending Score")
ax.legend()

st.pyplot(fig)

# Insights
st.subheader("💡 Business Insights")

for i in range(k):
    cluster = df[df["Cluster"] == i]
    avg_income = int(cluster["Annual_Income"].mean())
    avg_spend = int(cluster["Spending_Score"].mean())

    st.write(f"Cluster {i}: Avg Income = {avg_income}, Avg Spending = {avg_spend}")