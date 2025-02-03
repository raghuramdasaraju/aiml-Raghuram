#!/usr/bin/env python
# coding: utf-8

# In[1]:


#load the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv("NewspaperData.csv")
data


# In[3]:


data.info()


# In[4]:


data.isnull().sum()


# In[5]:


data.describe()


# In[6]:


#Boxplot for daily column
plt.figure(figsize=(6,3))
plt.title("Box plot for Daily Sales")
plt.boxplot(data["daily"], vert = False)
plt.show()


# In[7]:


sns.histplot(data['daily'], kde = True,stat='density',)
plt.show()


# In[8]:


sns.boxplot(x=data['sunday'])
plt.title("Boxplot of Sunday Data")
plt.show()


# #### Observations
# - There are no missing values
# - The daily column values appears to be right-skewed
# - The sunday column values also appear to be right-skewed
# - There are two outliers in both daily column and also in sunday column as observed from the boxplots

# #### Scatter plot and Correlation Strength

# In[9]:


x = data["daily"]
y = data["sunday"]
plt.scatter(data["daily"], data["sunday"])
plt.xlim(0, max(x) + 100)
plt.ylim(0, max(y) + 100)
plt.show()


# In[10]:


data["daily"].corr(data["sunday"])


# In[11]:


data[["daily","sunday"]].corr()


# #### Observations
# - The relationship between x (daily) and y (sunday) is seen to be linear as seen from scatter plot
# - The correlation is strong positive with Pearson's correlation coefficient of 0.958154

# #### Fit a Linear regression model

# In[12]:


# Build regression model

import statsmodels.formula.api as smf
model1 = smf.ols("sunday~daily",data = data).fit()


# In[13]:


model1.summary()


# In[14]:


x = data["daily"].values
y = data["sunday"].values
plt.scatter(x,y,color = "m",marker = "o", s = 30)
b0 = 13.84
b1 = 1.33
y_hat = b0 + b1*x
plt.plot(x,y_hat, color = "g")
plt.xlabel("x")
plt.ylabel("y")
plt.show()


# In[15]:


newdata = pd.Series([200,300,1500])


# In[16]:


data_pred=pd.DataFrame(newdata,columns=['daily'])
data_pred


# In[17]:


pred = model1.predict(data["daily"])
pred


# In[18]:


data["Y_hat"] = pred
data


# In[19]:


data["residuals"] = data["sunday"]-data["Y_hat"]
data


# In[20]:


import statsmodels.formula.api as smf
model1 = smf.ols("sunday~daily",data = data).fit()


# In[21]:


model1.summary()


# In[23]:


mse = np.mean((data["daily"]-data["Y_hat"])**2)
rmse = np.sqrt(mse)
print("MSE :",mse)
print("RMSE :",rmse)


# In[24]:


mae = np.mean(np.abs(data["daily"]-data["Y_hat"]))
mae


# |
