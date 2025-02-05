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

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import influence_plot
import numpy as np


# In[2]:


cars = pd.read_csv("Cars.csv")
cars.head()


# In[3]:


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

# In[4]:


cars.isnull().sum()


# Observations
# - There are no missing values.
# - There are 81 observations (81 different cars  data )
# - The data types of the columns are also relevent and valid 

# In[10]:


fig, (ax_box , ax_hist) = plt.subplots(2, sharex = True,gridspec_kw={"height_ratios": (.15, .85)})
sns.boxplot(data=cars , x = "HP" , ax= ax_box, orient="h")
ax_box.set(xlabel="")
sns.histplot(data=cars , x= "HP", ax =  ax_hist, bins=30, kde =True, stat= "density")
ax_hist.set(ylabel="Density")
plt.tight_layout()
plt.show()


# ### Observastions from boxplot and histograms
#   - There ome extreme values (outliers) observed in towards in towards the right tail of SP and HP distrubutions.
#   - In VOL and WT columns , a few outliers are observed in both tails of their distrubution.
#   - The extreme values of cars data may have comes from the specially designed nature of cars.
#   - As this is multi-dimensional data ,thr outlier with respect to spatial dimensions may have to be considered while building the regression mode

# ## Checking for duplicated rows

# In[12]:


cars[cars.duplicated()]


# ## pair plots ansd co-relation coefficient

# In[13]:


sns.set_style(style='darkgrid')
sns.pairplot(cars)


# In[14]:


cars.corr()


# ## Observations from correlation plots and coeffcients
#  - Betweem x and y ,all the x variables are showing moderate to high correlation strengths,hightest being between HP and MPG
#  - Therefore this dataseet gualifies for building a multiple linear regression model to predict MPG
#  - Among x columns (x1,x2,x3 and x4), some very high correlations strengths are observed between SP vs HP , VOL vs WT
#  - The high correlation among x columns is not desirable as it might lead to multicollinearity problem   

# ###  Preparing a preliminary model considering all X columns

# In[15]:


model = smf.ols("MPG~WT+VOL+SP+HP" ,data=cars).fit()


# In[16]:


model.summary()


# In[ ]:




