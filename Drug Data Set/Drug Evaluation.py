#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv('drug200.csv')
df.head()


# In[3]:


df['Sex'].value_counts()


# In[4]:


df['Sex']=df['Sex'].map({'F': 0, 'M':1})
df


# In[5]:


df['BP'].value_counts()


# In[6]:


df['BP']=df['BP'].map({'HIGH': '2', 'NORMAL':'1', 'LOW':'0'})
df['Cholesterol']=df['Cholesterol'].map({'HIGH': 1, 'NORMAL':0})


# In[7]:


df.head() 


# #logistic
# #KNN
# #SVM
# #DT
# #RANDOMFOREST
# #find test score

# In[8]:


X=df[['Age','Sex','BP','Cholesterol','Na_to_K']]
Y=df['Drug']


# In[9]:


df['Drug'].value_counts()


# In[10]:


df['Drug']=df['Drug'].map({'DrugY': 0, 'drugX':1, 'drugA':2, 'drugC':3, 'drugB':4})


# In[11]:


df


# ### Logistic Regression

# In[12]:


X=df[['Age','Sex','BP','Cholesterol','Na_to_K']]
Y=df['Drug']

from sklearn.model_selection import train_test_split
X_train, X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20, random_state=18)
#traintestsplit (random_state =18  test size=0.20)


# In[13]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(X_train)
X_train_scaled=scaler.transform(X_train)


# In[14]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_train_scaled,Y_train)
X_test_scaled = scaler.transform(X_test)
model.score(X_test_scaled,Y_test)


# ### KNN

# In[15]:


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5)


# In[16]:


model.fit(X_train_scaled, Y_train)


# In[17]:


X_test_scaled = scaler.transform(X_test)
model.score(X_test_scaled, Y_test)


# In[18]:


for i in range(1,10,2):
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(X_train_scaled,Y_train)
    score = model.score(X_test_scaled,Y_test)
    print('N:',i,'Score:',score)


# ### SVM

# In[19]:


from sklearn.svm import SVC
model_linear=SVC(kernel='linear')
model_linear.fit(X_train_scaled, Y_train)
model_linear.score(X_test_scaled, Y_test)


# In[20]:


from sklearn.svm import SVC
model_poly=SVC(kernel='poly')
model_poly.fit(X_train_scaled, Y_train)
model_poly.score(X_test_scaled, Y_test)


# In[21]:


from sklearn.svm import SVC
model_rbf=SVC(kernel='rbf')
model_rbf.fit(X_train_scaled, Y_train)
model_rbf.score(X_test_scaled, Y_test)


# ### DT

# In[22]:


from sklearn.tree import DecisionTreeClassifier


# In[23]:


model = DecisionTreeClassifier(random_state=100, max_depth=2)
model.fit(X_train,Y_train)
model.score(X_test,Y_test)


# In[24]:


model.score(X_train,Y_train)


# In[25]:


##GRIDSEARCH

from sklearn.model_selection import GridSearchCV


# In[26]:


params = {'max_depth':[1,2,3,4,5,6,7],'min_samples_leaf':[1,2,3,4,5],'min_samples_split':[3,4]}
grid_cv = GridSearchCV(DecisionTreeClassifier(random_state=100),param_grid=params,cv=4, verbose=2)
grid_cv.fit(X,Y)


# In[27]:


grid_cv.best_score_


# ### RandomForest

# In[28]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
random_cv= RandomizedSearchCV(RandomForestClassifier(random_state=100),param_distributions=params,cv=4, verbose=2)


# In[29]:


random_cv.fit(X,Y)


# In[30]:


random_cv.best_score_


# In[ ]:





# In[ ]:





# In[ ]:




