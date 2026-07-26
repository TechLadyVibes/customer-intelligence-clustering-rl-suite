# ============================================================
# CUSTOMER SEGMENTATION AND REINFORCEMENT LEARNING
# Unit 6 Programming Assignment
# Mohpheth Ekhaguere
# ============================================================

# ------------------------------------------------------------
# Import Libraries
# ------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score
)

from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage

# ============================================================
# CREATE CUSTOMER DATASET
# ============================================================

print("=" * 70)
print("CUSTOMER SEGMENTATION USING CLUSTERING ALGORITHMS")
print("=" * 70)

print("\nCreating Customer Dataset...\n")

customer_data = {

    "CustomerID": [
        "C1","C2","C3","C4","C5",
        "C6","C7","C8","C9","C10"
    ],

    "Age": [
        22,25,45,52,23,
        48,35,37,29,50
    ],

    "AnnualSpending": [
        200,220,800,850,250,
        780,400,420,300,900
    ],

    "PurchasesPerMonth": [
        5,6,2,1,7,
        2,4,3,5,1
    ]
}

df = pd.DataFrame(customer_data)

print("Dataset Created Successfully.\n")

# ============================================================
# DISPLAY DATASET
# ============================================================

print("=" * 70)
print("CUSTOMER DATASET")
print("=" * 70)

print(df)

print("\nDataset Shape")
print("----------------------------")

print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])

print("\nColumn Names")
print("----------------------------")

for column in df.columns:
    print("-", column)

print("\nMissing Values")
print("----------------------------")

print(df.isnull().sum())

print("\nSummary Statistics")
print("----------------------------")

print(df.describe())

# ============================================================
# DATA PREPARATION
# ============================================================

print("\n" + "=" * 70)
print("DATA PREPARATION")
print("=" * 70)

print("\nSelecting Features for Clustering...")

X = df[[
    "Age",
    "AnnualSpending",
    "PurchasesPerMonth"
]]

print("Features Selected Successfully.")

print("\nNormalizing Dataset...")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("Normalization Completed Successfully.")

print("\nScaled Dataset Shape :", X_scaled.shape)

# ============================================================
# ELBOW METHOD
# ============================================================

print("\n" + "=" * 70)
print("ELBOW METHOD")
print("=" * 70)

wcss = []

for k in range(1, 7):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    wcss.append(model.inertia_)

print("\nWCSS Values")

for i, value in enumerate(wcss, start=1):
    print(f"k = {i}   WCSS = {value:.4f}")

plt.figure(figsize=(7,5))

plt.plot(
    range(1,7),
    wcss,
    marker="o"
)

plt.title("Elbow Method")

plt.xlabel("Number of Clusters (k)")

plt.ylabel("WCSS")

plt.grid(True)

plt.show()

print("\nFrom the Elbow Method, k = 3 is selected.")

print("\nEND OF PART 1")
print("=" * 70)

# ============================================================
# K-MEANS CLUSTERING
# ============================================================

print("\n" + "=" * 70)
print("K-MEANS CLUSTERING")
print("=" * 70)

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

kmeans_labels = kmeans.fit_predict(X_scaled)

df["KMeansCluster"] = kmeans_labels

print("\nCluster Assignment")

print(df[
    [
        "CustomerID",
        "Age",
        "AnnualSpending",
        "PurchasesPerMonth",
        "KMeansCluster"
    ]
])

print("\nCluster Centers (Scaled Data)")
print(kmeans.cluster_centers_)

plt.figure(figsize=(7,6))

plt.scatter(
    X_scaled[:,1],
    X_scaled[:,2],
    c=kmeans_labels,
    s=100
)

plt.title("K-Means Customer Clusters")

plt.xlabel("Annual Spending (Scaled)")

plt.ylabel("Purchases Per Month (Scaled)")

plt.grid(True)

plt.show()


# ============================================================
# HIERARCHICAL CLUSTERING
# ============================================================

print("\n" + "=" * 70)
print("HIERARCHICAL CLUSTERING")
print("=" * 70)

linked = linkage(
    X_scaled,
    method="ward"
)

plt.figure(figsize=(8,5))

dendrogram(
    linked,
    labels=df["CustomerID"].values
)

plt.title("Hierarchical Clustering Dendrogram")

plt.xlabel("Customers")

plt.ylabel("Distance")

plt.grid(True)

plt.show()

hierarchical = AgglomerativeClustering(
    n_clusters=3
)

hierarchical_labels = hierarchical.fit_predict(X_scaled)

df["HierarchicalCluster"] = hierarchical_labels

print("\nHierarchical Cluster Assignment")

print(df[
    [
        "CustomerID",
        "HierarchicalCluster"
    ]
])

plt.figure(figsize=(7,6))

plt.scatter(
    X_scaled[:,1],
    X_scaled[:,2],
    c=hierarchical_labels,
    s=100
)

plt.title("Hierarchical Clustering Result")

plt.xlabel("Annual Spending (Scaled)")

plt.ylabel("Purchases Per Month (Scaled)")

plt.grid(True)

plt.show()


# ============================================================
# DBSCAN CLUSTERING
# ============================================================

print("\n" + "=" * 70)
print("DBSCAN CLUSTERING")
print("=" * 70)

dbscan = DBSCAN(
    eps=1.2,
    min_samples=2
)

dbscan_labels = dbscan.fit_predict(X_scaled)

df["DBSCANCluster"] = dbscan_labels

print("\nDBSCAN Cluster Assignment")

print(df[
    [
        "CustomerID",
        "DBSCANCluster"
    ]
])

plt.figure(figsize=(7,6))

plt.scatter(
    X_scaled[:,1],
    X_scaled[:,2],
    c=dbscan_labels,
    s=100
)

plt.title("DBSCAN Clustering Result")

plt.xlabel("Annual Spending (Scaled)")

plt.ylabel("Purchases Per Month (Scaled)")

plt.grid(True)

plt.show()


# ============================================================
# CLUSTER COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("COMPARISON OF CLUSTER RESULTS")
print("=" * 70)

comparison = df[
    [
        "CustomerID",
        "KMeansCluster",
        "HierarchicalCluster",
        "DBSCANCluster"
    ]
]

print(comparison)

print("\nObservations")
print("------------------------------------------")
print("1. K-Means partitions customers into")
print("   exactly three clusters.")
print()

print("2. Hierarchical clustering forms")
print("   clusters using a tree structure.")
print()

print("3. DBSCAN groups customers based")
print("   on density and may classify")
print("   isolated customers as noise (-1).")

print("\nEND OF PART 2")
print("=" * 70)

# ============================================================
# CLUSTERING PERFORMANCE EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("CLUSTERING PERFORMANCE EVALUATION")
print("=" * 70)

print("\nEvaluating Clustering Models...\n")

# -------------------------------
# K-Means Metrics
# -------------------------------

kmeans_silhouette = silhouette_score(
    X_scaled,
    kmeans_labels
)

kmeans_dbi = davies_bouldin_score(
    X_scaled,
    kmeans_labels
)

kmeans_wcss = kmeans.inertia_

print("K-Means")
print("----------------------------")
print(f"WCSS              : {kmeans_wcss:.4f}")
print(f"Silhouette Score  : {kmeans_silhouette:.4f}")
print(f"Davies-Bouldin    : {kmeans_dbi:.4f}")

# -------------------------------
# Hierarchical Metrics
# -------------------------------

hierarchical_silhouette = silhouette_score(
    X_scaled,
    hierarchical_labels
)

hierarchical_dbi = davies_bouldin_score(
    X_scaled,
    hierarchical_labels
)

print("\nHierarchical Clustering")
print("----------------------------")
print("WCSS              : Not Applicable")
print(f"Silhouette Score  : {hierarchical_silhouette:.4f}")
print(f"Davies-Bouldin    : {hierarchical_dbi:.4f}")

# -------------------------------
# DBSCAN Metrics
# -------------------------------

unique_clusters = set(dbscan_labels)

if len(unique_clusters - {-1}) >= 2:

    dbscan_silhouette = silhouette_score(
        X_scaled,
        dbscan_labels
    )

    dbscan_dbi = davies_bouldin_score(
        X_scaled,
        dbscan_labels
    )

    print("\nDBSCAN")
    print("----------------------------")
    print("WCSS              : Not Applicable")
    print(f"Silhouette Score  : {dbscan_silhouette:.4f}")
    print(f"Davies-Bouldin    : {dbscan_dbi:.4f}")

else:

    dbscan_silhouette = None

    print("\nDBSCAN")
    print("----------------------------")
    print("Not enough clusters were formed")
    print("to compute Silhouette Score.")
    print("Davies-Bouldin Index cannot")
    print("be calculated.")

# ============================================================
# PERFORMANCE COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("CLUSTER PERFORMANCE COMPARISON")
print("=" * 70)

print("\nInterpretation")
print("----------------------------------------")
print("Lower WCSS indicates more compact")
print("clusters for K-Means.")
print()

print("Higher Silhouette Score indicates")
print("better separated clusters.")
print()

print("Lower Davies-Bouldin Index")
print("indicates better clustering quality.")

best_method = "K-Means"

best_score = kmeans_silhouette

if hierarchical_silhouette > best_score:

    best_score = hierarchical_silhouette
    best_method = "Hierarchical Clustering"

if dbscan_silhouette is not None:

    if dbscan_silhouette > best_score:

        best_score = dbscan_silhouette
        best_method = "DBSCAN"

print("\nBest Clustering Method")
print("----------------------------")
print(best_method)

print("\nHighest Silhouette Score")
print(f"{best_score:.4f}")

# ============================================================
# REINFORCEMENT LEARNING CONCEPT
# ============================================================

print("\n" + "=" * 70)
print("REINFORCEMENT LEARNING EXAMPLE")
print("=" * 70)

print("\nScenario")
print("----------------------------------------")
print("The online store wants to learn")
print("the best discount strategy")
print("for each customer over time.")

print("\nAgent")
print("----------------------------------------")
print("AI Discount Recommendation System")

print("\nEnvironment")
print("----------------------------------------")
print("Online Retail Store")

print("\nPossible Actions")
print("----------------------------------------")
print("- Offer 0% Discount")
print("- Offer 5% Discount")
print("- Offer 10% Discount")
print("- Offer 20% Discount")

print("\nReward")
print("----------------------------------------")
print("Positive Reward:")
print("Customer purchases after")
print("receiving a discount.")
print()

print("Negative Reward:")
print("Customer ignores the offer")
print("or profit decreases.")

print("\nLearning Process")
print("----------------------------------------")
print("The agent repeatedly")
print("interacts with customers.")
print()

print("Successful discounts")
print("receive higher rewards.")
print()

print("Poor discount decisions")
print("receive lower rewards.")
print()

print("Over time, the system")
print("learns the discount")
print("strategy that maximizes")
print("long-term profit.")

# ============================================================
# LEARNING PARADIGM COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("LEARNING PARADIGM COMPARISON")
print("=" * 70)

print("\nSupervised Learning")
print("----------------------------------------")
print("Requires labeled data")
print("to predict known outputs.")

print("\nUnsupervised Learning")
print("----------------------------------------")
print("Finds hidden patterns")
print("without labeled data.")

print("\nReinforcement Learning")
print("----------------------------------------")
print("Learns through")
print("interaction with")
print("the environment")
print("using rewards and")
print("penalties.")

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print("Dataset Size           :", len(df))
print("Number of Features     :", X.shape[1])
print("Selected k             :", 3)
print("Best Method            :", best_method)
print("Best Silhouette Score  :", round(best_score, 4))

print("\nAssignment Completed Successfully.")

print("=" * 70)
print("END OF PROGRAM")
print("=" * 70)