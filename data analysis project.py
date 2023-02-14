#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
get_ipython().run_line_magic('matplotlib', 'inline')


# # Importing Data Sets

# In[3]:


file_name='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/kc_house_data_NaN.csv'


# In[5]:


df = pd.read_csv(file_name)
df.head()


# In[7]:


df.dtypes


# In[8]:


df.describe()


# # Data Wrangling

# In[9]:


df.dropna(subset=['id','Unnamed: 0'], axis=0, inplace = True)
df.describe()


# In[10]:


print("number of NaN values for the column bedrooms :",df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :",df['bathrooms'].isnull().sum())


# In[11]:


mean = df['bedrooms'].mean()
df['bedrooms'].replace(np.nan,mean, inplace = True)


# In[15]:


mean = df['bathrooms'].mean()
df['bathrooms'].replace(np.nan,mean, inplace = True)


# In[16]:


print("number of NaN values for the column bedrooms:",df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :",df['bathrooms'].isnull().sum())


# # Exploratory Data Analysis

# In[17]:


df['floors'].value_counts().to_frame()


# In[18]:


sns.boxplot(x='waterfront', y='price', data = df)


# In[19]:


sns.regplot(x = 'sqft_above', y = 'price', data = df)


# In[20]:


df.corr()['price'].sort_values()


# # Model devlopment

# In[21]:


x = df[['long']]
y = df[['price']]
lm = LinearRegression()
lm.fit(x,y)
lm.score(x,y)


# In[22]:


x = df[['sqft_living']]
y = df[['price']]
lm1 = LinearRegression()
lm1.fit(x,y)
lm1.score(x,y)


# In[23]:


features = df[['floors','waterfront','lat','bedrooms','sqft_basement','view','bathrooms','sqft_living15','sqft_above','grade','sqft_living']]


# In[24]:


lm1.fit(features,y)
lm1.score(features,y)


# In[25]:


Input=[('scale',StandardScaler()),('polynomial', PolynomialFeatures(include_bias=False)),('model',LinearRegression())]


# In[27]:


pipe = Pipeline(Input)
pipe.fit(x,y)
pipe.score(x,y)


# # Model Evaluation and Refinement

# In[28]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


# In[29]:


features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]

x = df[features]
y = df['price']
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.15, random_state = 1)


print('number of test samples:',x_test.shape[0])
print('number of training samples:',x_train.shape[0])


# In[30]:


from sklearn.linear_model import Ridge


# In[35]:


pr = PolynomialFeatures(degree = 2)
x_train_pr = pr.fit_transform(x_train[["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]])
x_test_pr = pr.fit_transform(x_test[["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]])

RigeModel = Ridge(alpha = 0.1)
RigeModel.fit(x_train,y_train)
RigeModel.score(x_train,y_train)


# In[36]:



RigeModel = Ridge(alpha = 0.1)
RigeModel.fit(x_test,y_test)
RigeModel.score(x_test,y_test)


# In[ ]:




