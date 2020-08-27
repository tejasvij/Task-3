#!/usr/bin/env python
# coding: utf-8

# # Classification Problem
# 
# Predicting the optimum number of clusters and representing it visually
# 
# Attribute Information:
# To construct the data, seven geometric parameters of Flower kernels were measured:
# 1.SepalLengthCm 2.SepalWidthCm 3.PetalLengthCm 4.PetalWidthCm 5.Species

# ## Solution Approach:
# 

# ### STEP 1:

# ##### Load the dataset

# In[203]:


import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[204]:


dataset = pd.read_csv('C:\\Users\\Tejasvi Jain\\Desktop\\Iris.csv',index_col='Id')


# ### STEP 2:
# 

# ##### Display the DataFrame to inspect the data

# In[205]:


dataset.head()


# ##### Exploring the data
# 

# In[206]:


dataset.shape


# In[207]:


dataset.describe()


# In[209]:


dataset.info


# ### Step 3:

# ##### Checking for any relationship between the data

# In[210]:


sns.pairplot(dataset, hue='Species')


# In[212]:


sns.heatmap(dataset.corr(),annot=True)


# ### Step 4:

# ### Elbow Method - To find optimal number of clusters for K-means
# 
# ##### Measure the quality of clusterings with different numbers of clusters using the inertia. For each of the given values of k, perform the following steps:
# 

# In[222]:


from sklearn.cluster import KMeans


# In[234]:


X=dataset.iloc[:,[0,1,2,3]].values


# In[224]:


inertias = []

for clusters in range(1,15):
    # Create a KMeans instance with k clusters: model
    kmeans = KMeans(n_clusters=clusters,init='k-means++')

    # Fit model to samples
    kmeans.fit(X)

    # Append the inertia to the list of inertias
    inertias.append(kmeans.inertia_)


# ### Step 5:
# 

# ##### Plot the inertia to see which number of clusters is best.
# 

# In[225]:


plt.plot(range(1,15), inertias, '-o')
plt.xlabel('number of clusters')
plt.ylabel('inertia')
plt.show()


# #### You can see from the graph that 3 is a good number of clusters, since these are points where the elbow begins

# ### Step 6:

# ### Training the model

# In[231]:


model = KMeans(n_clusters=3)


# In[232]:


labels = model.fit_predict(X)


# ### Step 7:

# ### Visualisation of Clusters

# In[233]:


plt.scatter(X[labels==0,0], X[labels==0,1], s=100, c='red', label='Iris-setosa')
plt.scatter(X[labels==1,0], X[labels==1,1], s=100, c='blue', label='Iris-versicolor')
plt.scatter(X[labels==2,0], X[labels==2,1], s=100, c='yellow', label='Iris-virginica')
plt.legend()


# In[ ]:




