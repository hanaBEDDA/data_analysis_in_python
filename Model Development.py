#!/usr/bin/env python
# coding: utf-8

# In[88]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[89]:


path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv'


# In[90]:


df = pd.read_csv(path)
df.head()


# #  Linear Regression and Multiple Linear Regression

# In[91]:


from sklearn.linear_model import LinearRegression


# In[92]:


lm = LinearRegression()


# In[93]:


x = df[['highway-mpg']]
y = df['price']


# In[94]:


lm.fit(x,y)


# In[95]:


#The prediction
yhat = lm.predict(x)
yhat[0:5]


# In[96]:


#The value of the intercept (a)
lm.intercept_


# In[97]:


#The value of the slope(b)
lm.coef_


# In[98]:


#Creat another linear regression object

lm1 = LinearRegression()


# In[99]:


lm1.fit(df[['engine-size']],df['price'])


# In[100]:


#the value of the intercept (a)
print(lm1.intercept_)

#The value of slope (b)
print(lm1.coef_)


# In[101]:


#The equation of prediction
yhat = -796.34 + 166.86*x

price = -796.34 + 166.86*df['engine-size']


# 
# Multiple Linear Regression

# In[102]:


z = df[['horsepower','curb-weight','engine-size','highway-mpg']]


# In[103]:


#Fit the model
lm.fit(z,df['price'])


# In[104]:


lm.intercept_


# In[105]:


lm.coef_


# # Model Evaluation Using Visualization

# In[106]:


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# Regression Plot

# In[107]:


width = 12
height = 10
plt.figure(figsize=(width,height))
sns.regplot(x='highway-mpg',y='price',data = df)
plt.ylim(0,)


# In[108]:


plt.figure(figsize=(width,height))
sns.regplot(x="peak-rpm",y='price',data = df)
plt.ylim(0,)


# In[109]:


df[['peak-rpm','highway-mpg','price']].corr()


# Residual Plot

# What is a residual?
# 
# The difference between the observed value (y) and the predicted value (Yhat) is called the residual (e). When we look at a regression plot, the residual is the distance from the data point to the fitted regression line.

# In[110]:


width = 12
height = 10
plt.figure(figsize=(width,height))
sns.residplot(x=df['highway-mpg'],y=df['price'])
plt.show()


# In[111]:


y_hat = lm.predict(z)


# In[112]:


plt.figure(figsize=(width,height))

ax1 = sns.distplot(df['price'], hist=False, color="r", label="Actual Value")
sns.distplot(y_hat, hist=False, color="b", label="Fitted Values" , ax=ax1)

plt.title('Actual vs Fitted Values for price ')
plt.xlabel('price(in dollars)')
plt.ylabel('Proportion of Cars')

plt.show()


# # Polynomial Regression and Pipelines

# In[113]:


#We will use this function to plot data
def PlotPolly(model, independent_variable, dependent_variable, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)
    
    plt.plot(independent_variable, dependent_variable, '.', x_new, y_new, '.')
    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')
    
    plt.show()
    plt.close()  


# In[114]:


x = df['highway-mpg']
y = df['price']


# In[115]:


#Here we use a polynomial of the 3rd order (cubic)
f = np.polyfit(x, y, 3)
p = np.poly1d(f)
print(p)


# In[116]:


PlotPolly(p, x, y, 'highway-mpg')


# In[117]:


np.polyfit(x,y,3)


# In[118]:


f1 = np.polyfit(x,y,11)
p1 = np.poly1d(f1)
print(p1)
PlotPolly(p1,x,y, 'Highway MPG')


# In[119]:


from sklearn.preprocessing import PolynomialFeatures


# In[120]:


pr = PolynomialFeatures(degree = 2)
pr


# In[121]:


Z_pr = pr.fit_transform(z)


# In[122]:


z.shape


# In[123]:


Z_pr.shape


# # Pipeline

# Data Pipelines simplify the steps of processing the data. We use the module Pipeline to create a pipeline. We also use StandardScaler as a step in our pipeline.

# In[124]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# In[125]:


Input=[('scale',StandardScaler()),('polynomial',PolynomialFeatures(include_bias=False)),('model',LinearRegression())]


# In[126]:


pipe = Pipeline(Input)
pipe


# In[127]:


z = z.astype(float)
pipe.fit(z,y)


# In[128]:


ypipe = pipe.predict(z)
ypipe[0:4]


# In[129]:


Input1 = [('scal',StandardScaler()),('model', LinearRegression())]

pipe = Pipeline(Input1)

pipe.fit(z,y)

ypipe = pipe.predict(z)
ypipe[0:4]


# # Measures for In-Sample Evaluation

# In[131]:


from sklearn.metrics import mean_squared_error


# In[133]:


mse = mean_squared_error(df['price'], yhat)
print('The mean square error of price and predicted value is :',mse)


# In[134]:


lm.fit(z, df['price'])

print('The R-square is: ',lm.score(z, df['price']))


# In[136]:


y_predict_multifit = lm.predict(z)
print('The mean square error of price and predicted value using multifit is :',      mean_squared_error(df['price'], y_predict_multifit))


# Polynomial Fit
# 

# In[137]:


from sklearn.metrics import r2_score


# In[139]:


r_squared = r2_score(y,p(x))
print("The R-square value is:",r_squared)


# In[140]:


mean_squared_error(df['price'],p(x))


# # Prediction and Decision  Making

# In[142]:


import matplotlib.pyplot as plt
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')


# In[143]:


new_input = np.arange(1,100,1).reshape(-1,1)


# In[145]:


lm = LinearRegression()


# In[150]:


x = df[['highway-mpg']]
y = df['price']


# In[152]:


lm.fit(x,y)


# In[154]:


yhat = lm.predict(new_input)
yhat[0:5]


# In[155]:


plt.plot(new_input,yhat)
plt.show()


# In[ ]:




