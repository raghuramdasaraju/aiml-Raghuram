#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


# In[2]:


iris = pd.read_csv("iris.csv")
iris


# In[3]:


import seaborn as sns
counts = iris["variety"].value_counts()
sns.barplot(data = counts)


# In[4]:


iris.info()


# In[5]:


iris[iris.duplicated(keep = False)]


# ## Observation 
# - There are 150 rows and 5 columns 
# - There are no Null values
# - There is one duplicated row 
# - The x-columns are sepal.length ,sepal.width,petal.length and petal.width
# - All the x-columns are continuous
# - The y-columns is "variety" which is categorical
# - There are three flower categories (classes)

# In[6]:


iris = iris.drop_duplicates(keep = "first")


# In[8]:


iris = iris.reset_index(drop=True)
iris


# #### Perform label encoding of target column

# In[9]:


labelencoder = LabelEncoder()
iris.iloc[:, -1] = labelencoder.fit_transform(iris.iloc[:,-1])
iris.head()


# In[10]:


iris.info()


# #### Observation 
# - The target column ('Variety') is still object type .it needs to be converted to numeric(int)

# In[12]:


iris['variety'] = pd.to_numeric(labelencoder.fit_transform(iris['variety']))
print(iris.info())


# In[13]:


X = iris.iloc[:,0:4]
Y = iris['variety']


# In[14]:


x_train,x_test,y_train,y_test = train_test_split(X,Y, test_size=0.3,random_state = 1)
x_train


# #### Building Decision Tree Classifier using Entropy Criteria

# In[16]:


model = DecisionTreeClassifier(criterion = 'entropy',max_depth = None)
model.fit(x_train,y_train)


# In[17]:


plt.figure(dpi = 1200)
tree.plot_tree(model)


# In[18]:


iris['variety'] = pd.to_numeric(labelencoder.fit_transform(iris['variety']))
print(iris.info())


# In[19]:


fn = ['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']
cn = ['setosa','versicolor','virginica']
plt.figure(dpi=1200)
tree.plot_tree(model,feature_names = fn,class_names=cn,filled = True)


# In[ ]:




