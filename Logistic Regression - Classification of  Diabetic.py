#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Classification of Diabitics using Logistic Regression  
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
data=pd.read_csv(r"D:\DON'T DELETE\Downloads\MLdataset\3.diabetes.csv")
print(data)


# In[9]:


X=data.values[:,0:8]
Y=data.values[:,8]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.33,random_state=10)
model=LogisticRegression(max_iter=3000)
model.fit(X_train,Y_train)
Y_pred=model.predict(X_test)


# In[10]:


print("Accuracy:",accuracy_score(Y_test,Y_pred)*100)


# In[15]:


print("confusion matrix:",confusion_matrix(Y_test,Y_pred))

       


# In[14]:


print ("Classification report:",classification_report(Y_test,Y_pred))


# In[ ]:




