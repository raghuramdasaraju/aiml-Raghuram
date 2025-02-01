#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[8]:


data1= pd.read_csv("NewspaperData .csv")
data


# In[9]:


plt.figure(figsize=(6,3))
plt.title("Box plot for Daily Sales")
plt.boxplot(data1["daily"], vert = False)
plt.show()


# In[11]:


sns.histplot(data1['daily'], kde = True,stat = "density",)
plt.show()


# In[12]:


sns.histplot(data1['sunday'], kde = True,stat = "density",)
plt.show()


# In[14]:


plt.figure(figsize=(6,3))
plt.title("Box plot for Daily Sales")
plt.boxplot(data1["sunday"], vert = False)
plt.show()


# # Observation

# ## . There are no  missing value
# ## . The daily column values appears to be right-skewed
# ## . The sunday column values also appear to be right-skewed
# ## . There are two outliers in both daily column and also in sunday column as observed from the boxplots

# In[16]:


x= data1["daily"]
y= data1["sunday"]
plt.scatter(data1["daily"], data1["sunday"])
plt.xlim(0, max(x) + 100)
plt.ylim(0, max(x) + 100)
plt.show()


# In[19]:


data1["daily"].corr(data1["sunday"])


# In[21]:


data1[["daily","sunday"]].corr()


# # Observations
# ## . The relationship between x(daily) and y(sunday) is seen to be linear as seen from scatter plot.
# ## . The correlation is strong positive wiyh pearson's correlation of 0.958154

# In[22]:


import statsmodels.formula.api as smf
model1 = smf.ols("sunday~daily",data = data1).fit


# In[23]:


model1.summary()

