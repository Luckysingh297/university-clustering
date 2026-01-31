"""CRISP-ML(Q) process model describes six phases:
1. Business and Data Understanding
2. Data Preparation
3. Model Building
4. Model Evaluation
5. Deployment
6. Monitoring and Maintenance

Problem Statements:
Global air travel has seen an upward trend in recent times. 
The maintenance of operational efficiency and maximizing profitability are crucial for airlines and airport authorities. 
Businesses need to optimize airline and terminal operations to enhance passenger satisfaction, improve turnover rates, and increase overall revenue. 
The airline companies with the available data want to find an opportunity to analyze and understand travel patterns, customer demand, and terminal usage.

Objective: Maximize the operational efficiency
Constraints: Maximize the financial health

Success Criteria: 
Business Success Criteria: Increase the operational efficiency by 10 % to 12% by segmenting the Airlines.
ML Success Criteria: Achieve a Silhouette coefficient of at least 0.7
Economic Success Criteria: The airline companies will see an increase in revenues by at least 8%(hypothetical numbers)

"""
# Data Understanding:
# Data Sources - Data Collection - Data Storage - 

#Data: The AirTraffic_Passenger_Statistics.csv
 
#Meta Data Description: : 
    #Activity Period : represents the period of activity
    #Operating Airline: This column specifies the name of the airline operating the flight.
    #Operating Airline IATA Code: This column contains the IATA (International Air Transport Association) code for the operating airline.
    #GEO Region: This column indicates the geographical region of the flight (e.g., US, Canada, Asia).
    #Terminal: This column specifies the terminal at the airport.
    #Boarding Area: This column denotes the boarding area within the terminal.
    #Passenger Count: This column represents the total number of passengers.
    #Year: This column indicates the year of the activity.
    #Month: This column provides the month of the activity.
    
import pandas as pd  
import seaborn as sns
import numpy as np  
import matplotlib.pyplot as plt 
import sweetviz
from sklearn.preprocessing import MinMaxScaler  
from sklearn.pipeline import make_pipeline  

from scipy.cluster.hierarchy import linkage, dendrogram  
from sklearn.cluster import AgglomerativeClustering  

from sklearn import metrics  
from clusteval import clusteval  
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine, text  
from urllib.parse import quote

#Reading datasets from an csv file into pandas datafram
airtraffic = pd.read_csv(r"C:\Users\LuckySingh\learn ML\assignments\assignment 2\Dataset\Dataset\AirTraffic_Passenger_Statistics.csv")

#Connect to database
user = "root"
pw = quote("cutelucky@575")
db = "airtraffic_db"
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

#to_sql : function to puse the datafram onto sql table (feed data into sql data base)
airtraffic.to_sql('airtraffic_table', con = engine, if_exists = 'replace', chunksize = 1000, index = False)


#(retrive the data from sql)
sql = 'select * from airtraffic_table;'
df = pd.read_sql_query(text(sql), engine.connect())

#data types
df.info()

#display the first few rows
df.head()

#descriptive statistics(EDA)
df.describe() 
df.shape
df.columns
# Check for missing values
print(df.isnull().sum())

import dtale

#Display the dataframr using dtale
d = dtale.show(df, host = 'localhost', port = '8000')

#open to browser to view intractive D-Tale Dashbord
d.open_browser()

#Data preprocessing
# **Cleaning Unwanted columns**
#Activity Period is identify to a numerical code combining year and month (e.g., 200507 = July 2005).
#it's does not use of Activity Period and Operating Airline IATA Codecolumns
#Droping the Activity Period, Operating Airline IATA Code columns
df.drop(['Activity Period'], axis = 1, inplace = True)
df.drop(['Operating Airline IATA Code'], axis = 1, inplace = True)
df.info()

#EDA report highlighths:

#clean columns
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
print(df.columns.tolist())    
df.select_dtypes(include=[np.number]).columns.tolist()

#Select only numeric columns
cols = ['passenger_count', 'year']  
df_num = df[cols]


# Normalization/MinMax Scaler - To address the scale differences

# Creating a pipeline using make_pipeline to apply MinMaxScaler for feature scaling
pipe1 = make_pipeline(MinMaxScaler())

# Train the data preprocessing pipeline on data
# Applying the pipeline 'pipe1' to transform the cleaned DataFrame 'df' and storing the transformed data in a new DataFrame 'df_pipelined'
df_pipelined = pd.DataFrame(pipe1.fit_transform(df_num), columns = cols, index = df_num.index)

# Displaying the first few rows of the transformed DataFrame 'df_pipelined' to inspect the changes
df_pipelined.head()

# Generating descriptive statistics of the transformed DataFrame 'df_pipelined'
# The scale of the data is normalized to have a minimum value of 0 and a maximum value of 1 due to MinMaxScaler
df_pipelined.describe()

#Save Preprocessed scaled data into SQL Mandatory
user = 'root' #user name
pw = quote('cutelucky@575')# password
db = 'airtraffic_db' # database
engine = create_engine(f'mysql+pymysql://{user}:{pw}@localhost/{db}')
df_pipelined.to_sql('airtraffic_scaled', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

print(df_pipelined.isnull().sum())

######### Model Building #########
# CLUSTERING MODEL BUILDING

# Hierarchical Clustering - Agglomerative Clustering
#plt.figure(1, figsize = (16, 8))# creating a new figure with specified size for the dendrogram plot

#genrating a dendrogram plot using hierachical clustring with complete linkage
#tree_plot = dendrogram(linkage(df_pipelined, method = "complete"))
Z = linkage(df_pipelined.iloc[:50], method='complete')
plt.figure(1, figsize = (16, 8))
dendrogram(Z)
plt.title('Hierachical clustring Dendrogram')#setting the titel of the dendrogram plot
plt.xlabel('Index') #setting the lable for x-axis
plt.ylabel('Encludian distance')#setting the lable for y-axis
plt.show()# Displaying the dendrogram plot

# Applying AgglomerativeClustering and grouping data into 3 clusters 
# based on the above dendrogram as a reference
# Creating an instance of AgglomerativeClustering with parameters: 
hc1 = AgglomerativeClustering(n_clusters = 3, metric = 'euclidean', linkage = 'complete')

#Fitting the agglomerativeclustering model to data predicting the cluster lable for each sample
y_hc1 = hc1.fit_predict(df_pipelined)

# Displaying the cluster labels assigned by the AgglomerativeClustering model
y_hc1

# Accessing the cluster labels directly from the AgglomerativeClustering model
hc1.labels_ 

#converting the cluster label into padas series for further analysis
cluster_labels = pd.Series(hc1.labels_)

#combine the label obtained with data
#concatanating the cluster labels with clean dataframe(df)alog the colums axis
df_clust = pd.concat([cluster_labels, df], axis = 1)

#displaying the first fews rows of dataframe
df_clust.head()

#Displaying the clomns name
df_clust.columns

#renaming the first columns (containing cluster labels) to 'cluster' for better clarity
df_clust = df_clust.rename(columns = {0: 'cluster'})

#displaying first few rows of DataFrame after renaming of first column
df_clust.head()

#cluster evaluation

# Silhouette coefficient:
# Silhouette coefficient is a Metric, which is used for calculating 
# goodness of the clustering technique, and the value ranges between (-1 to +1).
# It tells how similar an object is to its own cluster (cohesion) compared to 
# other clusters (separation).
# A score of 1 denotes the best meaning that the data point is very compact 
# within the cluster to which it belongs and far away from the other clusters.
# Values near 0 denote overlapping clusters.

silhouette = metrics.silhouette_score(df_pipelined, cluster_labels)
print(silhouette)
# **Calinski Harabasz:**
from sklearn.metrics import calinski_harabasz_score
calinski_harabasz = calinski_harabasz_score(df_pipelined, cluster_labels)
print(calinski_harabasz)

# **Davies-Bouldin Index:**
davies_bouldin = metrics.davies_bouldin_score(df_pipelined, cluster_labels) # Lower DB scores indicate better clustering (closer to 0 is ideal).
print(davies_bouldin)


#Deployment: Save clustered data into SQL
df_clust = df.copy()
df_clust['cluster'] = cluster_labels
df_clust.to_sql('airtraffic_clustered', con=engine, if_exists='replace', chunksize=1000, index=False)

print("Clustered dataset successfully saved into 'airtraffic_clustered' table.")

# ---------------- Monitoring Function ----------------
def run_clustering(engine, table_name="airtraffic_table", n_clusters=3, threshold=0.7):
    sql = f"SELECT * FROM {table_name};"
    df = pd.read_sql_query(text(sql), engine.connect())

    # Drop unwanted columns
    df = df.drop(['Activity Period', 'Operating Airline IATA Code'], axis=1, errors='ignore')
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    # Numeric columns
    cols = ['passenger_count', 'year']
    df_num = df[cols]

    # Scale
    pipe = make_pipeline(MinMaxScaler())
    df_scaled = pd.DataFrame(pipe.fit_transform(df_num), columns=cols, index=df.index)

    # Clustering
    hc = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='complete')
    cluster_labels = hc.fit_predict(df_scaled)
    df['cluster'] = cluster_labels

    # Save to SQL
    df.to_sql('airtraffic_clustered', con=engine, if_exists='replace', chunksize=1000, index=False)

    # Evaluation metrics
    silhouette = metrics.silhouette_score(df_scaled, cluster_labels)
    calinski = calinski_harabasz_score(df_scaled, cluster_labels)
    db_index = metrics.davies_bouldin_score(df_scaled, cluster_labels)

    print("Monitoring run complete.")
    print(f"Silhouette Score: {silhouette:.3f}")
    print(f"Calinski-Harabasz Index: {calinski:.3f}")
    print(f"Davies-Bouldin Index: {db_index:.3f}")

    if silhouette < threshold:
        print(f" Alert: Silhouette score {silhouette:.2f} dropped below threshold {threshold}. Consider retraining.")

    return df, silhouette, calinski, db_index



