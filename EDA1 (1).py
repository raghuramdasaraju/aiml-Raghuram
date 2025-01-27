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


# In[15]:


cols = data1.columns
colours = ['orange','green']
sns.heatmap(data1[cols].isnull(),cmap=sns.color_palette(colours),cbar=True)


# In[16]:


median_ozone = data1["Ozone"].median()
mean_ozone = data1["Ozone"].mean()
print('median',median_ozone)
print('mean',mean_ozone)


# In[17]:


data1['Ozone'] = data1['Ozone'].fillna(median_ozone)
data1.isnull().sum()


# In[18]:


print(data1["Weather"].value_counts())
mode_weather = data1["Weather"].mode()[0]
print(mode_weather)


# In[19]:


data1["Weather"]= data1["Weather"].fillna(mode_weather)
data1.isnull().sum()


# In[20]:


# Impute missing values (Replace NaN with  mode etc.) of "month" using fillna()
mode_month = data1["Month"].mode()[0]
data1["Month"] = data1["Month"].fillna(mode_month)
data1.isnull().sum()


# In[21]:


data1.tail()


# In[26]:


fig, axes = plt.subplots(2,1,figsize=(8,6),gridspec_kw={'height_ratios': [1,3]})
sns.boxplot(data=data1["Ozone"],ax=axes[0],color = 'skyblue',width=0.5,orient = 'h')
axes[0].set_title("Boxplot")
axes[0].set_xlabel("Ozone Levels")
sns.histplot(data1["Ozone"],kde=True,ax=axes[1],color='purple',bins=30)
axes[1].set_title("Histogram with KDE")
axes[1].set_xtitle("Ozone Levels")
axes[1].set_ytitle("Frequency")
plt.tight_layout()
plt.show()


# In[23]:


sns.violinplot(data=data1["Ozone"],color='lightgreen')
plt.title("Vilolin plot")


# In[25]:


sns.violinplot(data=data1["Solar"],color='lightblue')
plt.title("Vilolin plot")


# In[27]:


fig, axes = plt.subplots(2,1,figsize=(8,6),gridspec_kw={'height_ratios': [1,3]})
sns.boxplot(data=data1["Solar"],ax=axes[0],color = 'skyblue',width=0.5,orient = 'h')
axes[0].set_title("Boxplot")
axes[0].set_xlabel("Solar Levels")
sns.histplot(data1["Solar"],kde=True,ax=axes[1],color='purple',bins=30)
axes[1].set_title("Histogram with KDE")
axes[1].set_xtitle("Solar Levels")
axes[1].set_ytitle("Frequency")
plt.tight_layout()
plt.show()


# In[ ]:




