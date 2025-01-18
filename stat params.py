#!/usr/bin/env python
# coding: utf-8

# In[1]:


def mean_value(L):
    A = sum(L)
    B = len(L)
    avg = A/B
    return avg
mean_value([34,6,67,89,89])


# In[2]:


def mode_value(L):
    s = set(L)
    d = {}
    for x in s :
        d[x] = L.count(x)
    m = max(d.values())
    for k in d.keys():
        if d[k] == n:
            return k
    


# In[ ]:


list = []

