#!/usr/bin/env python
# coding: utf-8

# In[6]:


#Import liabraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[7]:


# Load Dataset
url = "http://bit.ly/w-data"
stud_data = pd.read_csv(url)


# In[8]:


#quick Review
stud_data.head(10)


# In[9]:


stud_data.shape


# In[10]:


stud_data.plot(kind='scatter',x='Scores',y='Hours')
plt.show()


# In[13]:


#Dividing the data
X = stud_data.iloc[:, :-1].values 
Y = stud_data.iloc[:, 1].values 


# In[15]:


#Splitting of data in training and test set
from sklearn.model_selection import train_test_split 
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)


# In[16]:


#Training the model
from sklearn.linear_model import LinearRegression
lm= LinearRegression()
lm.fit(X_train,Y_train)


# In[17]:


line = lm.coef_*X+lm.intercept_


# In[19]:


#plotting scatter plot for the test data
plt.scatter(X, y)
plt.plot(X,line)
plt.show()


# In[23]:


#Predicting the model
print(X_test)
Y_pred =lm.predict(X_test) # Predicting the scores
print(Y_pred)


# In[22]:


# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_pred})  
df 


# In[25]:


#Predict for own values

Hours=[[9.25]]
Scores_predict=lm.predict(Hours)
print(Scores_predict)


# In[27]:


#Evaluating the model
from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, Y_pred)) 

