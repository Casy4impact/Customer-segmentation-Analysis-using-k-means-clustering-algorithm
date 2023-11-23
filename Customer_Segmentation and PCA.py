#!/usr/bin/env python
# coding: utf-8

# # Project Overview:
# ## Objective: 
# To enhance Acme Stores' online retail strategy, this project aims to perform an Exploratory Analysis and implement Customer Segmentation Analysis. The goal is to identify the most valuable customer segment, enabling Acme to optimize resource allocation and maximize return on investment (ROI).
# ________________________________________
# ## Problem Overview: 
# Acme Stores, a UK-based online retail store specializing in unique occasion gifts, seeks to elevate customer satisfaction and increase ROI by delivering personalized experiences and tailored recommendations. Currently, the lack of insights into customer marketing behavior hinders the effective targeting and engagement of the right customer segments.
# ________________________________________
# ## Objectives:
# - 1.	Conduct an Exploratory Analysis to understand customer marketing behavior.
# - 2.	Implement Customer Segmentation Analysis to identify the most valuable customer segment.
# - 3.	Redistribute company resources based on the segmented data.
# - 4.	Develop personalized marketing strategies for each identified customer segment.
# - 5.	Improve targeting precision to increase sales and overall ROI.
# 

# In[ ]:


#conda install yellowbrick


# ### **UNSUPERVISED MACHINE LEARNING**
# Naturally in Machine Learning, We often implemement UnSupervised ML models as either; a part of the Feature Engineering Process,i.e to either help us obtain more insights during the EDA, or as a step to reduce the dimensions to improve the performance of any Supervised ML model. In this project, I will be implememnting it as part of the feature engineering process, in order to aid us obtaim more insights from our data.
# 
# For this this project, I will use the pca for feature decomposition and kmeans as a clustering algorithm. Although, there are other decomposition algortihms and other clustering algorithms which are not tested in this project
# 

# **Note**
# 
# Data Visualization and EDA Skills will be used to Carry out Univariate, Bivariate and Multivariate Analysis EDA for this Data Set

# In[ ]:


#import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np

import warnings
warnings.filterwarnings("ignore")


# ### ASSESING DATA

# In[ ]:


# Load the data
data = pd.read_csv("data.csv",encoding='UTF-8')
data.head()


# In[ ]:


data.info()


# In[ ]:


data.describe().T


# In[ ]:


data.describe(exclude=['int64','float64']).T


# In[ ]:


#identifying cancelled orders
data.query("Quantity < 0")


# In[ ]:


#identifying missing features
data.isna().sum()


# In[ ]:


#identifying duplicated data points
data.duplicated().sum()


# #### observations
# **After assessing our data set, we have outlined a couple of problems with the dataset;**
# - wrong datatype, the invoice date should be a datetime object
# - presence of missing data points, the customer id is an important feature, product description feature has some missing data points
# - data duplicates, we have identified about 5268 data observations as duplicates
# - we also have quite a numbe of cancelled orders, this sets up an interesting area that should be investigated

# In[ ]:


#change datetime from string to datetime object
data.InvoiceDate = pd.to_datetime(data.InvoiceDate)


# In[ ]:


#drop off missing datapoints
data.dropna(inplace=True)


# In[ ]:


#remove data duplicates
data.drop_duplicates(inplace=True)


# In[ ]:


#filter out cancelled orders
data = data.query("Quantity > 0")


# In[ ]:


# creating the total sales feature
data["SalePrice"] = data["Quantity"] * data["UnitPrice"]


# In[ ]:


data.head(2)


# ### Feature Engineering

# In[ ]:


data.shape


# In[ ]:


#grouping our data by the customer id
cust_data = data.groupby("CustomerID")


# In[ ]:


# calculate the total sales, order_count, and the average order value per customer
totalSales = cust_data["SalePrice"].sum()
order_count = cust_data["InvoiceDate"].size()
avg_order_value = totalSales / order_count

data2 = pd.DataFrame({
    "TotalSales":totalSales,
    "OrderCount":order_count,
    "AvgOrdVal":avg_order_value
})

data2.head(2)


# In[ ]:


data2.shape


# In[ ]:


#visualize the total sales feature
plt.figure(figsize=(5,2))
g = sb.boxplot(data=data2,x='TotalSales');
plt.title("total sales")
plt.show()

sb.histplot(data2.TotalSales,bins=100);


# In[ ]:





# ## Data normalization
# 
# Clustering algorithms (k-means clustering including) are highly affected by the scales of the data. So we need to [normalize](https://en.wikipedia.org/wiki/Normalization_(statistics) the data to be on the same scale.

# In[ ]:


#normalize data
from sklearn.preprocessing import MinMaxScaler, StandardScaler,RobustScaler

scaler = MinMaxScaler()
scaled_df = pd.DataFrame(scaler.fit_transform(data2),index = data2.index, columns=data2.columns)
scaled_df


# In[ ]:





# # UNSUPERVISED ML SECTION

# ## **Dimensionality Reduction**
#  More input features often make a predictive modelling task more challenging to model, more generally referred to as the curse of dimensionality. thus,Dimensionality reduction refers to techniques that reduce the number of input variables in a dataset.
# 
# Principal Componenet Analysis (PCA) is a technique for redcuing the dimensions of a large dataset, increasing the interpreatblilty and at the same minimizing information loss
# other exmaples of Dimensionality Reduction techniques include Self Organizing Maps (SOM), t-distributed Stochastic Neighbor Embedding (t-SNE) etc.

# In[ ]:


#import pca from sklearn lib
from sklearn.decomposition import PCA

#instantiate pca
pca = PCA(n_components=2)

pca_df = pd.DataFrame(pca.fit_transform(scaled_df),columns=(["pca1","pca2"]))


# In[ ]:


#exploring our components
pca_df.head(2)


# In[ ]:


pca.explained_variance_ratio_


# In[ ]:


#visualizing our new data dimensions
x = pca_df['pca1']
y = pca_df['pca2']
#z = pca_df['pca3']

fig = plt.figure(figsize=(10,7))
ax=fig.add_subplot(111,projection="3d")
ax.scatter(x,y,marker="o")
ax.set_title("3d visualization of our new dimensions")


# In[ ]:





# **We will use our new dimensions from the PCA to train our clustering model and segment our customer base**

# 
# ## **K-means clustering**
# 
# **K-means clustering** is a method of vector quantization, originally from signal processing, that aims to **partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean** ([Wiki](https://en.wikipedia.org/wiki/K-means_clustering)). This is a method of **unsupervised learning** that learns the commonalities between groups of data without any target or labeled variable.
# 
# K-means clustering algorithm spits the records in the data into a **pre-defined number of clusters**, where the data points within each cluster are close to each other. One difficulty of using k-means clustering for customer segmentation is the fact that you need to know the number of clusters beforehand. Luckily, the silhouette coefficient can help you.
# 
# **The silhouette coefficient** measures how close the data points are to their clusters compared to other clusters. The silhouette coefficient values range from -1 to 1, where the closer the values are to 1, the better they are.
# 
# Let's find the best number of clusters:

# In[ ]:


from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans

elbow = KElbowVisualizer(estimator=KMeans())
elbow.fit(pca_df)


# ### APPLYING CLUSTERING ALGO

# In[ ]:


#instantiating clustering model
kmeans = KMeans(n_clusters=5)

#fitting model on pca components and creating clusters
y_means = kmeans.fit_predict(pca_df)

#appending new clusters to data
data2["clusters"] = y_means


# ### CLUSTERING MODEL EVALUATION
# There isn't a sure or single most appropriate way to evaluate a clustering algorithm, as this can vary by the nature of problem being solved, the domain of that problem, and the business metric of the domain under study.
# In this project, the Silhouette score is adopted in evaluating the model's performance. 
# There are other metrics/methods for evaluating clustering algortihms.
# > The silhouette coefficient measures how close the data points are to their clusters compared to other clusters. The silhouette coefficient values range from -1 to 1, where the closer the values are to 1, the better they are.
# 
# 
# ### Here are some metrics and methods for evaluating clustering algorithms:
# 
# - 1. Silhouette coefficient: 
# The silhouette coefficient measures how well a point is clustered within its cluster. It ranges from -1 to 1, where a value of -1 indicates that a point is more likely to belong to another cluster, and a value of 1 indicates that a point is well-clustered.
# - 2. Calinski-Harabasz index: 
# The Calinski-Harabasz index measures the ratio of the within-cluster variance to the between-cluster variance. A higher index indicates better clustering.
# - 3. Dunn index: 
# The Dunn index measures the ratio of the smallest inter-cluster distance to the largest intra-cluster distance. A higher index indicates better clustering.
# - 4. Davies-Bouldin index: 
# The Davies-Bouldin index measures the average silhouette coefficient of all clusters. A lower index indicates better clustering.
# - 5. Gap statistic: 
# The gap statistic compares the within-cluster variance of a clustering algorithm to the expected within-cluster variance of a random clustering. A lower gap statistic indicates better clustering.
# - 6. F-measure: 
# The F-measure is a combination of precision and recall. Precision is the proportion of points in a cluster that actually belong to that cluster, while recall is the proportion of points that belong to a cluster that are actually included in that cluster. A higher F-measure indicates better clustering.
# Rand index: The Rand index measures the similarity between two clusterings. It ranges from 0 to 1, where a value of 0 indicates that the two clusterings are completely different, and a value of 1 indicates that the two clusterings are identical.
# 
# ##### In addition to these metrics, there are a number of other methods that can be used to evaluate clustering algorithms. These include:
# 
# - Visualizing the clusters: 
# This can be done by plotting the points in a two- or three-dimensional space. The clusters should be visible as distinct groups of points.
# - Examining the cluster sizes: 
# The clusters should be of a reasonable size. Clusters that are too small or too large may indicate that the clustering algorithm is not working well.
# - Examining the cluster distributions: 
# The clusters should be well-separated in the feature space. Clusters that are too close together may indicate that the clustering algorithm is not working well.
# 
# The choice of metric or method will depend on the specific application. For example, if the goal of clustering is to identify groups of similar customers for marketing purposes, then the silhouette coefficient or the Calinski-Harabasz index might be appropriate metrics. However, if the goal of clustering is to identify groups of genes that are co-regulated, then the F-measure or the Rand index might be more appropriate.
# 
# It is also important to note that no single metric or method is perfect. Therefore, it is often a good idea to use a combination of metrics and methods to evaluate a clustering algorithm.

# In[ ]:


#import silhouette score from sklearn
from sklearn.metrics import silhouette_score

# Calculate the silhouette score
silhouette_score = silhouette_score(data2,
    data2["clusters"], metric="euclidean")
print(f"Silhouette Score: {silhouette_score:.4f}")


# **we have a silhouette score of 0.72, though not the best of scores, but its close to 1, so we will use for this business case**

# In[ ]:





# ### Now Carry EDA post Feature Engineering
# - This is where the power of clustering is more felt and useful, we have created segmented our customers, we will leverage on our new segments/clusters and carry out a post EDA to now derive insights and answer laid out business objectives

# In[ ]:


#exploring distributions of clusters
sb.countplot(x="clusters",data=data2)


# In[ ]:


#distribution of customers accross the 5 new clusters
data2.clusters.value_counts()


# In[ ]:


data2.columns


# In[ ]:





# In[ ]:


#visualizing our new data dimensions
x = data2['TotalSales']
y = data2['OrderCount']
z = data2['AvgOrdVal']
cmap = "Accent"

fig = plt.figure(figsize=(10,7))
ax=fig.add_subplot(111,projection="3d")
ax.scatter(x,y,z,c=data2.clusters,marker="o",cmap=cmap)
ax.set_xlabel("total sales")
ax.set_ylabel("order counts")
ax.set_zlabel(" average order value")
ax.set_title("3d visualization of our new dimensions")
plt.show()


# In[ ]:





# In[ ]:


#relationship between order count vs average order value vs customer clusters
plt.figure(figsize=(10,5))
plt.scatter(
    data2["AvgOrdVal"],
    data2["OrderCount"],
    c = data2["clusters"],
    s = 50,
    cmap = "Accent"

)
plt.title("k-means clstering")
plt.xlabel("AvgOrdVal")
plt.ylabel("OrderCount")
plt.show()


# In[ ]:


#relationship between order count vs total sales vs customer clusters
plt.figure(figsize=(10,5))
plt.scatter(
    data2["TotalSales"],
    data2["OrderCount"],
    c = data2["clusters"],
    s = 50,
    cmap = "Accent"

)
plt.title("k-means clstering")
plt.xlabel("TotalSales")
plt.ylabel("OrderCount")
plt.show()


# In[ ]:


#total sales vs customer clusters
sb.barplot(y="TotalSales",x="clusters",data=data2);
plt.title("total sales vs clusters")


# In[ ]:


#average order value vs clusters
sb.barplot(y="AvgOrdVal",x="clusters",data=data2)


# In[ ]:


#order count vs clusters
sb.boxplot(y="OrderCount",x="clusters",data=data2)


# **observations**
# - clusters 1 and cluster 2 apears to be our group of our high value customers, cluster one don't order regularly, but they spend more when they eventually do, while cluster 2 customers, spend more (not as high as cluster 1 customers), and they also order regulary, so these customers are the top valuable customers for the business.
# - customers in cluster 4 are our average customers, these guys order regulary, though not as regular as the cluster 2 customers, and they tend to spend more too, though not as high as cutomes in clusters 1 and 2.
# - customers in clusters 0 are our low value customers, they don't have a high order rate and they don't spend much either
# - cluster 3 segment have the highest order count, so these customers are ordering  regularly  but they don't purchase expensive products or spend on expensive items at the store

# ### RECOMMENDATIONS AND CONCLUSION
# 

# After delving into the intricacies of our customer segmentation analysis using unsupervised learning with k-means clustering algorithm, I am excited to present targeted recommendations designed to elevate our customer base, drive profitability, and refine our strategic approach.
# 1. Cluster 1 - Occasional High-Spenders:
# •	Recommendation: Launch exclusive promotions or limited-time offers to entice Cluster 1. Encourage their occasional high-value purchases by creating a sense of urgency, making them feel privileged as premium customers.
# 2. Cluster 2 - Regular and Valuable:
# •	Recommendation: Establish a loyalty program for Cluster 2, rewarding their regular orders. Provide incentives for consistent purchases, enhancing their loyalty and potentially increasing their average spend over time.
# 3. Cluster 4 - Average Yet Valuable:
# •	Recommendation: Implement personalized product recommendations for Cluster 4 based on their regular orders and spending patterns. Encourage them to explore additional offerings, potentially boosting both order frequency and average spending.
# 4. Cluster 0 - Low-Value Engagement:
# •	Recommendation: Launch targeted re-engagement campaigns for Cluster 0. Offer special discounts, limited-time promotions, or exclusive deals to reignite their interest and elevate their engagement with our brand.
# 5. Cluster 3 - Frequent Orders, Lower Spend:
# •	Recommendation: Introduce bundled deals or tiered discounts for Cluster 3. Encourage them to explore higher-value products with incentives tied to their frequent orders, maximizing both order count and average spend.
# 6. Cross-Cluster Promotions:
# •	Recommendation: Craft cross-cluster promotions that appeal to overlapping preferences. For instance, offer Cluster 2 customers an exclusive deal when they refer a customer from Cluster 4, fostering a sense of community and broadening our customer base.
# 7. Enhanced Customer Experience for All:
# •	Recommendation: Invest in an improved online shopping experience. Streamline the purchasing process, implement personalized product recommendations, and provide excellent customer service to elevate the satisfaction of all customer clusters.
# 8. Targeted Ad Campaigns:
# •	Recommendation: Develop targeted ad campaigns for each cluster, emphasizing the unique value propositions that cater to their preferences. Leverage social media, email, and online platforms to ensure our messaging reaches the right audience at the right time.
# 9. Data-Driven Inventory Management:
# •	Recommendation: Optimize inventory based on the preferences of each cluster. Ensure that popular products among high-value clusters are well-stocked, while also experimenting with new offerings to capture the interest of other segments.
# 10. Regular Analysis and Adaptation:
# •	Recommendation: Establish a routine for continuous analysis of customer segmentation data. Adapt strategies based on evolving trends and customer behaviors, ensuring that our approach remains dynamic and responsive.
# 
# Implementing these recommendations will not only fortify our relationships with existing customers but also create pathways to attract new customers. The key lies in our ability to understand, adapt, and tailor our strategies to the diverse needs of each customer cluster.
# Should you have any questions or wish to discuss these strategies further, I am at your disposal.
# Best regards,
# ## Contact info
# Anyaegbu, Casmir Ndukaku
# casmiranyaegbu@gmail.com
# +2348030581387
# 
# 

# In[ ]:




