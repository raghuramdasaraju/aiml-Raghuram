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

# In[5]:


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

# In[6]:


cars[cars.duplicated()]


# ## pair plots ansd co-relation coefficient

# In[7]:


sns.set_style(style='darkgrid')
sns.pairplot(cars)


# In[8]:


cars.corr()


# ## Observations from correlation plots and coeffcients
#  - Betweem x and y ,all the x variables are showing moderate to high correlation strengths,hightest being between HP and MPG
#  - Therefore this dataseet gualifies for building a multiple linear regression model to predict MPG
#  - Among x columns (x1,x2,x3 and x4), some very high correlations strengths are observed between SP vs HP , VOL vs WT
#  - The high correlation among x columns is not desirable as it might lead to multicollinearity problem   

# ###  Preparing a preliminary model considering all X columns

# In[9]:


model = smf.ols("MPG~WT+VOL+SP+HP" ,data=cars).fit()


# In[10]:


model.summary()


# In[11]:


#Find the performance metrics
#Create a data frame with actual y and predicted y columns
df1 = pd.DataFrame()
df1["actual_y1"] = cars["MPG"]
df1.head()


# In[12]:


#Predict for the given X data columns
pred_y1 = model.predict(cars.iloc[:,0:4])
df1["pred_y1"] = pred_y1
df1.head()


# In[13]:


# Compute the mean squared Error (MSE), RMSE for model1
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df1["actual_y1"], df1["pred_y1"])
print("MSE :", mse)
print("RMSE :",np.sqrt(mse))


# In[14]:


cars.head()


# In[15]:


# Compute VIF values
rsq_hp = smf.ols('HP~WT+VOL+SP',data=cars).fit().rsquared
vif_hp = 1/(1-rsq_hp)

rsq_wt = smf.ols('WT~HP+VOL+SP',data=cars).fit().rsquared  
vif_wt = 1/(1-rsq_wt) 

rsq_vol = smf.ols('VOL~WT+SP+HP',data=cars).fit().rsquared  
vif_vol = 1/(1-rsq_vol) 

rsq_sp = smf.ols('SP~WT+VOL+HP',data=cars).fit().rsquared  
vif_sp = 1/(1-rsq_sp) 

# Storing vif values in a data frame
d1 = {'Variables':['Hp','WT','VOL','SP'],'VIF':[vif_hp,vif_wt,vif_vol,vif_sp]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame


# ## Observation :
# - The ideal range of VIF values shall be between 0 to 10 However slightly higher values can be tolerated.
# - As seen from the very high VIF values for VOL and WT,it is clear that they are prone to multicollinearity  problem 
# - Hence it is decided to drop one of the columns (either VOL or WT) to overcome the multicollinearity.
# - It is decided to drop WT and retain VOL column in further models

# In[16]:


cars1 = cars.drop("WT" , axis=1)
cars1.head()


# In[17]:


import statsmodels.formula.api as smf
model2 = smf.ols("MPG~VOL+SP+HP",data=cars1).fit()


# In[18]:


model2.summary()


# In[19]:


df2 = pd.DataFrame()
df2["actual_y2"] = cars["MPG"]
df2.head()


# In[20]:


pred_y2 = model2.predict(cars.iloc[:,0:4])
df2["pred_y2"] = pred_y2
df2.head()


# In[21]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df2["actual_y2"], df2["pred_y2"])
print("MSE :", mse)
print("RMSE :",np.sqrt(mse))


# ## Observations from model2 summary
# - The adjusted R-suared value improved slightly to 0.76
# - All the p-values for model parameters  are less than 5% hence they are significant
# - Therefore the HP,VOL,SP columns are finalized as the significant predictor for the MPG response varible
# - There is no improvement in MSE value

# In[22]:


cars1


# In[26]:


### Identification of High influence
cars1.shape


# # Leverage (Hat Values):
# Leverage values diagnose if a data point has an extreme value in terms of the independent variables. A point with high leverage has a great ability to influence the regression line. The threshold for considering a point as having high leverage is typically set at 3(k+1)/n, where k is the number of predictors and n is the sample size.

# In[23]:


cars1[cars1.index.isin([65,70,76,78,79,80])]


# In[24]:


cars2=cars1.drop(cars1.index[[65,70,76,78,79,80]],axis=0).reset_index(drop=True)


# In[25]:


cars2


# ### Build Model3 on cars2 dataset

# In[30]:


model3= smf.ols("MPG~VOL+SP+HP",data = cars2).fit()


# In[31]:


model3.summary()


# ### Performance Metrics for model3

# In[33]:


df3 = pd.DataFrame()
df3["actual_y3"] = cars2["MPG"]
df3.head()


# In[34]:


pred_y3 = model3.predict(cars2.iloc[:,0:3])
df3["pred_y3"] = pred_y3
df3.head()


# In[37]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df3["actual_y3"],df3["pred_y3"])
print("MSE :", mse)
print("RMSE :",np.sqrt(mse))


# #### Comparison of models
#                      
# 
# | Metric         | Model 1 | Model 2 | Model 3 |
# |----------------|---------|---------|---------|
# | R-squared      | 0.771   | 0.770   | 0.885   |
# | Adj. R-squared | 0.758   | 0.761   | 0.880   |
# | MSE            | 18.89   | 18.91   | 8.68    |
# | RMSE           | 4.34    | 4.34    | 2.94    |
# 
# 
# - **From the above comparison table it is observed that model3 is the best among all with superior performance metrics**
# 
# 

# In[ ]:





# In[ ]:




