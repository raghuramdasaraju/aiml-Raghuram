#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv("data_clean.csv")
data


# In[3]:


data.info()


# In[4]:


data.shape


# In[5]:


data.dtypes


# In[6]:


data.describe()


# In[7]:


print(type(data))
print(data.shape)


# In[8]:


data1 = data.drop(['Unnamed: 0',"Temp C"],axis = 1)
data1


# In[9]:


data1.info


# In[10]:


data1["Month"]=pd.to_numeric(data['Month'],errors='coerce')
data1.info()


# In[11]:


data1[data1.duplicated()]


# In[12]:


data1.rename({'Solar.R' : 'Solar'}, axis=1 ,inplace = True)
data1


# In[13]:


data1.info()


# In[14]:


data1.isnull().sum()


# In[20]:


cols = data1.columns
colours = ['orange','green']
sns.heatmap(data1[cols].isnull(),cmap=sns.color_palette(colours),cbar=True)


# In[26]:


median_ozone = data1["Ozone"].median()
mean_ozone = data1["Ozone"].mean()
print('median',median_ozone)
print('mean',mean_ozone)


# In[24]:


data1['Ozone'] = data1['Ozone'].fillna(median_ozone)
data1.isnull().sum()


# In[ ]:




