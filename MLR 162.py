#!/usr/bin/env python
# coding: utf-8

# Assumptions in Multilinear Regression
# 
#  1. Linearity : The relationship between the predictors and the response is linear.
# 
#  2. Independence : Observations are independent of each other.
# 
#  3. Homoscedasticity : The residuals (Y - Y_hat)) exhibit constant variance at all levels of the predictor.
# 
#  4. Normal Distribution of Errors : The residuals of the model are normally distributed.
# 
#  5. No multicollinearity : The independent variables should not to be too highly corelated with each other.

# In[8]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import influence_plot
import numpy as np


# In[11]:


cars = pd.read_csv("Cars.csv")
cars.head()


# In[12]:


cars = pd.DataFrame(cars, columns=["HP","VOL","SP","WT","MPG"])
cars.head()


# Description of columns
# . MPG : Milege of the car (Mile per Gallon) (This is Y-column to be predicted)
# 
# . HP : Horse Power of the cars(X1 columns)
# 
# . VOL : Volume of the car (size)(X2 columns)
# 
# . SP : Top speed of the car (Miles per Hour)(X3 column)
# 
# . WT : Weight of the car (Pounds)(X4 Column)

# In[14]:


cars.isnull().sum()


# Observations
# - There are no missing values.
# - There are 81 observations (81 different cars  data )
# - The data types of the columns are also relevent and valid 

# In[ ]:




