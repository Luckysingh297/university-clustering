# Step 2: Data Understanding

import pandas as pd

# Load dataset
df = pd.read_excel(r"C:\Users\LuckySingh\learn ML\assignments\assignment 2\Dataset\Dataset\University_Clustering.xlsx")

# Show top 5 rows
print("🔹 Sample Data:")
print(df.head())

# Show info to check datatypes and null values
print("\n🔹 Dataset Info:")
print(df.info())

# Summary statistics
print("\n🔹 Summary:")
print(df.describe())

# Step 3: Data Preparation


from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1️⃣ Drop irrelevant columns
df_clean = df.drop(columns=["UnivID", "Univ"])

# 2️⃣ Handle missing values (fill with mean)
num_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns
df_clean[num_cols] = df_clean[num_cols].fillna(df_clean[num_cols].mean())

# 3️⃣ Encode categorical column "State"
le = LabelEncoder()
df_clean['State'] = le.fit_transform(df_clean['State'])

# 4️⃣ Scaling - very important for distance-based clustering
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_clean)

# Convert to DataFrame for readability
scaled_df = pd.DataFrame(scaled_data, columns=df_clean.columns)

print("✅ Data Cleaning Done! Here’s your prepared data:")
print(scaled_df.head())


import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch

# Create dendrogram using Ward linkage (minimizes variance)
plt.figure(figsize=(10, 6))
dendrogram = sch.dendrogram(sch.linkage(scaled_df, method='ward'))
plt.title("Dendrogram for Hierarchical Clustering")
plt.xlabel("Universities")
plt.ylabel("Euclidean Distance")
plt.show()



from sklearn.cluster import AgglomerativeClustering

# Updated: replace affinity with metric
hc = AgglomerativeClustering(n_clusters=4, metric='euclidean', linkage='ward')

# Fit model and predict clusters
clusters = hc.fit_predict(scaled_df)

# Add cluster labels to original dataset
df['Cluster'] = clusters

print("✅ Hierarchical Clustering completed successfully!")
print(df[['Univ', 'State', 'Cluster']].head(10))

from sklearn.metrics import silhouette_score

# Evaluate clustering quality
score = silhouette_score(scaled_df, df['Cluster'])
print(f"🔹 Silhouette Score: {score:.3f}")





import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
sns.scatterplot(
    data=df,
    x='SAT',
    y='Expenses',
    hue='Cluster',
    palette='viridis',
    s=100
)
plt.title("🎯 Hierarchical Clustering Results (SAT vs Expenses)")
plt.xlabel("SAT Score")
plt.ylabel("Expenses ($)")
plt.legend(title="Cluster")
plt.show()


# Save the dataset with clusters
df.to_excel("University_Hierarchical_Clusters.xlsx", index=False)
print("💾 Saved as: University_Hierarchical_Clusters.xlsx")
















