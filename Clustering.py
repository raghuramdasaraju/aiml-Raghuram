#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans


# ### Clustering - Divide the universities in to groups (clusters)

# In[10]:


Univ = pd.read_csv("Universities.csv")
Univ


# In[11]:


Univ1 = Univ.iloc[:,1:]


# In[12]:


Univ1


# In[13]:


cols = Univ1.columns


# In[14]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_Univ_df = pd.DataFrame(scaler.fit_transform(Univ1),columns = cols)
scaled_Univ_df


# In[ ]:




