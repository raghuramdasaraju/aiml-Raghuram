#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
import numpy as np


# In[27]:


df = pd.read_csv("Universities.csv")
df


# In[28]:


np.mean(df["SAT"])


# In[29]:


np.median(df["SAT"])


# In[30]:


df.describe()


# In[31]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[32]:


plt.figure(figsize=(6,3))
plt.title("Acceptence Ratio")
plt.hist(df['Accept'])


# In[33]:


plt.figure(figsize=(9,7))
plt.title("Acceptence Ratio")
plt.hist(df['Accept'])


# In[34]:


sns.histplot(df["Accept"])


# In[35]:


sns.histplot(df["Accept"],kde = True)


# In[36]:


sns.histplot(df["SFRatio"],kde = True)


# In[37]:


sns.histplot(df["Expenses"],kde = True)


# In[40]:


s1 = [1,23,45,67,89,89,888]
scores1 = pd.Series(s1)
scores1


# In[41]:


plt.boxplot(scores1, vert = False)


# In[ ]:





# In[ ]:




