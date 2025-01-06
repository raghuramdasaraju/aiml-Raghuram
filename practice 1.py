#!/usr/bin/env python
# coding: utf-8

# In[3]:


list1 = ["A","B","C","D"]
my_dict = dict.fromkeys(list1,20)
my_dict


# In[4]:


list1 = ["A","B","C","D"]
my_dict = dict.fromkeys(list1,10)
my_dict


# In[6]:


scores = {"virat":10,
         "virat1":90,
         "virat2":80,
         "virat3":50,
         "virat4":30,}
print(scores)


# In[10]:


scores_list = []
for v in scores.values():
    scores_list.append(v)
print(scores_list)
print("Total Score: ",sum(scores_list))


# In[ ]:




