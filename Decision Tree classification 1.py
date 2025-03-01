#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


# In[12]:


iris = pd.read_csv("iris.csv")
iris


# In[13]:


import seaborn as sns
counts = iris["variety"].value_counts()
sns.barplot(data = counts)


# In[14]:


iris.info()


# In[15]:


iris[iris.duplicated(keep = False)]


# ## Observation 
# - There are 150 rows and 5 columns 
# - There are no Null values
# - There is one duplicated row 
# - The x-columns are sepal.length ,sepal.width,petal.length and petal.width
# - All the x-columns are continuous
# - The y-columns is "variety" which is categorical
# - There are three flower categories (classes)

# In[17]:


iris = iris.drop_duplicates(keep = "first")


# In[19]:


iris = iris.reset_index(drop=True)
iris


# In[ ]:




