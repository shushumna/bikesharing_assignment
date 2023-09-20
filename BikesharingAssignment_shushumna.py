#!/usr/bin/env python
# coding: utf-8

# The steps that we are doing on the data set are:
# 
# 1.Data understanding and visualizing
# 2.Preparing data for modeling
# 3.Training the model
# 4.Buliding a linear model
# 5.Residual Analysis

# # step 1:Data understanding and visualizing

# In[1]:


# importing the required libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import warnings

import statsmodels
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression


# In[2]:


# reading the data

day=pd.read_csv("day.csv")


# In[3]:


#checking the present data in the dataset 

day.head()


# In[4]:


#checking the number of rows and coloumns in the dataset

day.shape


# In[5]:


#To find missing values and information about the dataset

day.info()


# In[6]:


#describing the columns of the given dataset

day.describe()


# In[7]:


# renaming some columns for better understanding

day.rename(columns={'yr':'year','mnth':'month','hum':'humidity'},inplace=True)


# In[8]:


#check the datasheet

day.head()


# In[9]:


#creating new dataframe and copying present dataframe into new one

day1=day.copy()


# In[10]:


#checking the shape of new dataframe

day1.shape


# In[11]:


# removing duplicates if any present in the dataframe

day1.drop_duplicates(inplace=True)


# In[12]:


day1.shape


# The above shape is same as the original dataframe shape so that means there are no duplicates in the original dataframe

# In[13]:


#visualizing the data

sns.pairplot(data=day,vars=['temp','atemp','humidity','windspeed','cnt'])
plt.show()


# In[14]:


#visualizing the catagorical variables

plt.figure(figsize=(20,12))
plt.subplot(2,4,1)
sns.boxplot(x='season',y='cnt',data=day)
plt.subplot(2,4,2)
sns.boxplot(x='month',y='cnt',data=day)
plt.subplot(2,4,3)
sns.boxplot(x='weekday',y='cnt',data=day)
plt.subplot(2,4,4)
sns.boxplot(x='weathersit',y='cnt',data=day)
plt.subplot(2,4,5)
sns.boxplot(x='holiday',y='cnt',data=day)
plt.subplot(2,4,6)
sns.boxplot(x='workingday',y='cnt',data=day)
plt.subplot(2,4,7)
sns.boxplot(x='year',y='cnt',data=day)
plt.show()


# # Step 2: Preparing data for modeling

# In[15]:


# dropping some unwanted columns
# cnt column having both casual and registered columns data we are dropping them
# dteday data is present in both month and year except for day so we are dropping them

day.drop(['instant','dteday','casual','registered'],axis=1,inplace=True)


# In[16]:


# mapping the season column

day.season=day.season.map({1:'spring',2:'summer',3:'fall',4:'winter'})


# In[17]:


# mapping the month column

day.month=day.month.map({1:'Jan',2:'Feb',3:'Mar',4:'April',5:'May',6:'June',7:'July',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'})


# In[18]:


# mapping the weekday column

day.weekday=day.weekday.map({0:'Sunday',1:'Monday',2:'Tuesday',3:'Wednesday',4:'Thursday',5:'Friday',6:'Saturday'})


# In[19]:


# mapping weathersit column

day.weathersit=day.weathersit.map({1:'Clear',2:'mist+cloudy',3:'Light snow',4:'Heavy Rain + Ice Pallets + Thunderstorm + Mist'})


# In[20]:


day.head()


# In[21]:


day.shape


# In[22]:


# dummy variable creation for categorical variables they are month, weekday, weathersit and season variables

months_day=pd.get_dummies(day.month,drop_first=True)
weekday_day=pd.get_dummies(day.weekday,drop_first=True)
weathersit_day=pd.get_dummies(day.weathersit,drop_first=True)
season_day=pd.get_dummies(day.season,drop_first=True)


# In[23]:


day.head()


# In[24]:


# merging the dataframes with dummy variables

day_new=pd.concat([day,months_day,weekday_day,weathersit_day,season_day],axis=1)


# In[25]:


day_new.head()


# In[26]:


day_new.info()


# In[27]:


# dropping unnecessary columns as we already created dummy variables

day_new.drop(['season','month','weekday','weathersit'],axis=1,inplace=True)


# In[28]:


#checking the data

day_new.head()


# In[29]:


#checking the no of rows and columns in the modified dataframe

day_new.shape


# In[30]:


#checking information of dataset

day_new.info()


# #### Splitting the data into Training and Testing sets

# In[31]:


#splitting the dataframe into train and test

df_train, df_test=train_test_split(day_new, train_size=0.7, random_state=100)


# In[32]:


df_train.shape


# In[33]:


df_test.shape


# ## Rescalling the features

# In[34]:


#using MinMaxScaler for recalling the features

scaler=MinMaxScaler()


# In[35]:


#Apply scaler() to all columns except dummy variables

num_vars=['temp','atemp','humidity','windspeed','cnt']
df_train[num_vars]=scaler.fit_transform(df_train[num_vars])


# In[36]:


df_train.head()


# # Step 3: Training the model

# In[37]:


#heatmap

plt.figure(figsize=(25,25))
sns.heatmap(df_train.corr(), annot=True, cmap="YlGnBu")
plt.show()


# In[38]:


df_train.head()


# In[39]:


# Building linear model

y_train=df_train.pop('cnt')
x_train=df_train


# In[40]:


lm=LinearRegression()
lm.fit(x_train, y_train)

rfe = RFE(lm)
rfe = rfe.fit(x_train, y_train)


# In[41]:


list(zip(x_train.columns,rfe.support_,rfe.ranking_))


# In[42]:


col=x_train.columns[rfe.support_]
col


# In[43]:


x_train.columns[~rfe.support_]


# In[44]:


# Create a dataframe that will contain the names of all the feature variables 
def calculateVIF(day):
    vif=pd.DataFrame()
    vif['Features']=day.columns
    vif['VIF']=[variance_inflation_factor(day.values,i) for i in range(day.shape[1])]
    vif['VIF']=round(vif['VIF'], 2)
    vif=vif.sort_values(by="VIF",ascending= False)
    return vif


# In[45]:


x_train_rfe=x_train[col]


# In[46]:


calculateVIF(x_train_rfe)


# # Step 4:Building a linear model

# In[47]:


#Building 1st linear regression model

x_train_lm_l=sm.add_constant(x_train_rfe)
lr_l=sm.OLS(y_train,x_train_lm_l).fit()
lr_l.summary()


# In[48]:


x_train_new=x_train_rfe.drop(['humidity'],axis=1)
calculateVIF(x_train_new)


# In[49]:


#Building 2nd linear regression model

x_train_lm_2=sm.add_constant(x_train_new)
lr_2=sm.OLS(y_train,x_train_lm_2).fit()
lr_2.summary()


# In[50]:


#Building 3rd linear regression model

x_train_lm_3=sm.add_constant(x_train_new)
lr_3=sm.OLS(y_train,x_train_lm_3).fit()
lr_3.summary()


# In[51]:


x_train_new=x_train_new.drop(['holiday'],axis=1)


# In[52]:


calculateVIF(x_train_new)


# In[53]:


#Building 4th linear regression model

x_train_lm_4=sm.add_constant(x_train_new)
lr_4=sm.OLS(y_train,x_train_lm_4).fit()
lr_4.summary()


# In[54]:


x_train_new=x_train_new.drop(['July'],axis=1)


# In[55]:


calculateVIF(x_train_new)


# In[56]:


#Building 5th linear regression model

x_train_lm_5=sm.add_constant(x_train_new)
lr_5=sm.OLS(y_train,x_train_lm_5).fit()
lr_5.summary()


# # Step 5:Residual Analysis

# In[57]:


y_train_cnt=lm.predict(x_train)


# In[58]:


#plot the histogram of the error terms

fig=plt.figure()
sns.displot((y_train - y_train_cnt), bins=20)
fig.suptitle('Error Terms',fontsize=20)
plt.xlabel('Errors',fontsize=18)


# In[59]:


num_vars=['temp','atemp','humidity','windspeed','cnt']
df_test[num_vars]=scaler.transform(df_test[num_vars])


# In[60]:


#Dividing into x_test and y_test

y_test=df_test.pop('cnt')
x_test=df_test


# In[61]:


#creating x_test_new dataframe by dropping variables from x_test
x_test_new=x_test[x_train_new.columns]

#Adding a constant variable
x_test_new=sm.add_constant(x_test_new)


# In[63]:


#making predictions
y_pred=lm.predict(x_test)


# ## Model Evaluation

# In[65]:


#plotting y_test and y_pred
fig=plt.figure()
plt.scatter(y_test,y_pred)
fig.suptitle('y_test vs y_pred',fontsize=20)
plt.xlabel('y_test',fontsize=18)
plt.ylabel('y_pred',fontsize=16)

