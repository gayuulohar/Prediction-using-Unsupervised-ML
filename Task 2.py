#!/usr/bin/env python
# coding: utf-8

# # Sparks Foundation GRIP - May 2023
# 
# # Role : The Data science and Business Analytics Intern
# 
# # Author : Gayatri Lohar
# 
# # Task 2 : Prediction using unsupervised ML
# 
# # Problem Statement : Predict the optimun number of clusters and represent it visually.
# 
# # Dataset link= https://bit.ly/3kXTdox

# # Step 1: Importing the modules & dataset

# In[1]:


import pandas as pd     #for data manipulation and analysis
import numpy as np      #to work with arrays
import seaborn as sns   #for data visualization and exploratory data analysis
import matplotlib.pyplot as plt  #for data visualization and plotting framework
from sklearn import datasets   #for k-means clustering of data


# In[2]:


data=pd.read_csv(r'C:\Users\gayat\OneDrive\Documents\Data\Iris.csv')
data.head()


# # Step 2: Getting the information of the data

# In[3]:


data.shape


# In[4]:


data.info()


# In[5]:


data.columns


# In[6]:


data['Species'].unique()


# In[7]:


data.describe()


# In[8]:


#Checks for duplicate
data.duplicated().sum()


# In[9]:


#To find the null values in the data
data.isnull().sum()


# In[10]:


#now we will drop the label column because it is on unsupervised learning model
iris = pd.DataFrame(data)
iris_data = iris.drop(columns = ['Species','Id'])
iris_data.head()


# # Step 3: Find the Optimal numbers of clusters using the Elow method. 
#         So in Elbow method:- The number of clusters are varied with the certain range. For each numbers, within cluster sum of square (wss) value will be calculated and stored in a list.These value are plotted against the range of numbers of clusters used before.

# In[14]:


# Finding the optimum number of clusters for k-means classification
x = data.iloc[:,[0,1,2,3]].values
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters= i,init= 'k-means++', 
                    max_iter= 300, n_init= 10, random_state= 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)


# In[15]:


#Allowing us to observe 'The elbow'

plt.plot(range(1, 11), wcss)
plt.title('The Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')           
plt.show()


# In[16]:


# Applying kmeans to the dataset

kmeans = KMeans(n_clusters = 3, init = 'k-means++',
                max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)
y_kmeans


# # Step 4: Visualising the test set result

# In[17]:


plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], 
            s = 30, c = 'blue', label = 'Iris-setosa')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], 
            s = 30, c = 'red', label = 'Iris-versicolour')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1],
            s = 30, c = 'yellow', label = 'Iris-virginica')

# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 100, c = 'green', label = 'Centroids')
plt.legend()
plt.grid()
plt.show()


# # Thank You.
