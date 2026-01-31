# 🏫 University Clustering using Hierarchical Clustering

![Python](https://img.shields.io/badge/Python-3.x-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

## 📌 Project Overview
This project focuses on **clustering universities** based on their academic performance and financial factors using **Hierarchical Clustering (Agglomerative Clustering)**.  
The goal is to group universities into meaningful clusters and analyze the characteristics of each cluster.

---

## 🎯 Objectives
✅ Perform Exploratory Data Analysis (EDA)  
✅ Clean and preprocess the dataset  
✅ Apply feature scaling for distance-based clustering  
✅ Build a **Hierarchical Clustering Model**  
✅ Visualize clusters using **Dendrogram** and **PCA**  
✅ Evaluate clusters using **Silhouette Score**  
✅ Export final clustered data for reporting

---

## 📂 Dataset Information
The dataset includes universities with the following columns:

- **UnivID** → Unique University ID  
- **Univ** → University Name  
- **State** → State/Region  
- **SAT** → SAT score  
- **Top10** → % students from Top 10%  
- **Accept** → Acceptance rate  
- **SFRatio** → Student-Faculty ratio  
- **Expenses** → Annual expenses  
- **GradRate** → Graduation rate  

---

## 🧠 Workflow
### ✅ 1. Data Understanding
- Loaded data from Excel file  
- Checked sample rows (`head()`)  
- Verified datatypes and missing values (`info()`)  
- Statistical summary (`describe()`)

---

### ✅ 2. Data Preprocessing
- Dropped irrelevant columns: `UnivID`, `Univ`  
- Handled missing values using **mean imputation**  
- Encoded categorical variable: `State`  
- Scaled features using **StandardScaler** *(very important for clustering)*

---

### ✅ 3. Dendrogram Visualization
A dendrogram was created using **Ward linkage** to identify optimal cluster formation.

---

### ✅ 4. Model Building (Agglomerative Clustering)
Hierarchical clustering was performed using:

- Linkage: **Ward**
- Distance metric: **Euclidean**
- Number of clusters: **4**

---

### ✅ 5. Model Evaluation
Used **Silhouette Score** to evaluate clustering quality.

📌 **Silhouette Score:** `0.247`

---

### ✅ 6. Cluster Insights
Cluster-wise average values:

| Cluster | SAT (Avg) | Top10 (Avg) | Expenses (Avg) | GradRate (Avg) |
|--------:|----------:|------------:|---------------:|---------------:|
| 0 | 1362.78 | 90.56 | 41176.89 | 92.22 |
| 1 | 1061.50 | 38.75 | 9953.00 | 71.75 |
| 2 | 1247.50 | 76.71 | 22473.71 | 85.29 |
| 3 | 1282.00 | 81.00 | 23396.00 | 91.50 |

---

### ✅ 7. PCA Visualization
Clusters were visualized in 2D using **PCA (Principal Component Analysis)**.

📍 PCA plot shows clear separation for some clusters and slight overlap for mid-range universities.

---

## 🛠️ Tech Stack
- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Matplotlib**
- **Seaborn**
- **SciPy**
