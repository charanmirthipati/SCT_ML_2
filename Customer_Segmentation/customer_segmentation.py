import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv("Mall_Customers.csv")
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# K-Means Clustering

kmeans = KMeans(
    n_clusters=5,
    init='k-means++',
    random_state=42
)

y_kmeans = kmeans.fit_predict(X)

df['Cluster'] = y_kmeans

print("\nCustomers in each Cluster:")
print(df['Cluster'].value_counts())

print("\nCluster Statistics:")
print(df.groupby('Cluster').mean(numeric_only=True))

# Customer Segmentation Graph

plt.figure(figsize=(12,8))

plt.scatter(
    X.iloc[y_kmeans == 0, 0],
    X.iloc[y_kmeans == 0, 1],
    s=70,
    label='Cluster 1'
)

plt.scatter(
    X.iloc[y_kmeans == 1, 0],
    X.iloc[y_kmeans == 1, 1],
    s=70,
    label='Cluster 2'
)

plt.scatter(
    X.iloc[y_kmeans == 2, 0],
    X.iloc[y_kmeans == 2, 1],
    s=70,
    label='Cluster 3'
)

plt.scatter(
    X.iloc[y_kmeans == 3, 0],
    X.iloc[y_kmeans == 3, 1],
    s=70,
    label='Cluster 4'
)

plt.scatter(
    X.iloc[y_kmeans == 4, 0],
    X.iloc[y_kmeans == 4, 1],
    s=70,
    label='Cluster 5'
)

plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=400,
    c='black',
    marker='X',
    label='Centroids'
)

plt.title("Customer Segmentation using K-Means Clustering")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.grid(True)

plt.show()

# Business Insights

print("\nBusiness Insights")
print("-" * 50)

print("Cluster 1 -> Premium Customers")
print("Cluster 2 -> Potential Customers")
print("Cluster 3 -> Regular Customers")
print("Cluster 4 -> Budget Customers")
print("Cluster 5 -> Low Priority Customers")

print("\n" + "=" * 60)
print("MANAGEMENT SUMMARY")
print("=" * 60)

print("""
1. The customers were divided into 5 segments using K-Means Clustering.

2. Premium Customers:
   High Income + High Spending
   -> Most valuable customer group.

3. Potential Customers:
   High Income + Low Spending
   -> Can be converted into premium customers.

4. Regular Customers:
   Average Income + Average Spending
   -> Stable revenue source.

5. Promotion-Responsive Customers:
   Lower income but higher spending.
   -> Respond well to discounts.

6. Recommendation:
   Focus marketing efforts on Premium and Potential Customers
   to maximize revenue and customer retention.
""")