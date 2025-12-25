"""
Customer Segmentation using KMeans Clustering

This script groups customers into different segments based on
their purchasing behavior using the Mall Customers dataset.
"""

# ---------------------------
# 1. Import Required Libraries
# ---------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ---------------------------
# 2. Load the Dataset
# ---------------------------
DATA_PATH = r"C:\Users\lenovo\PycharmProjects\Customer_Segmentation\Mall_Customers.csv"

df = pd.read_csv(DATA_PATH)

# ---------------------------
# 3. Select Relevant Features
# ---------------------------
# Using income and spending score for customer behavior analysis
features = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# ---------------------------
# 4. Scale the Features
# ---------------------------
# KMeans works on distances, so scaling is mandatory
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# ---------------------------
# 5. Elbow Method to Find Optimal K
# ---------------------------
wcss = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method for Optimal Number of Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# ---------------------------
# 6. Apply KMeans Clustering
# ---------------------------
# From the elbow graph, k = 5 is optimal
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(scaled_features)

# ---------------------------
# 7. Visualize Customer Segments
# ---------------------------
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    hue='Cluster',
    data=df,
    palette='Set2',
    s=100
)

plt.title("Customer Segmentation using KMeans")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend(title="Cluster")
plt.show()

# ---------------------------
# 8. Cluster Characteristics
# ---------------------------
cluster_summary = df.groupby('Cluster')[[
    'Annual Income (k$)',
    'Spending Score (1-100)'
]].mean()

print("\nCustomer Segment Characteristics:\n")
print(cluster_summary)
