#!/usr/bin/env python
# coding: utf-8

# ### Imorting Libraries

# In[39]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import seaborn as sns


# ### Importing dataset

# In[2]:


forest_data=pd.read_csv('forestfires.csv')
forest_data


# ### Data understanding

# In[3]:


forest_data.dtypes


# In[4]:


forest_data.isna().sum()


# In[5]:


forest_data.head()


# ### Data preprocessing

# In[9]:


#Dropping the month and day columns
forest_data=forest_data.drop(labels=['month','day'], axis=1)
forest_data


# In[41]:


forest_data.dtypes


# ### Model Building

# In[11]:


X=forest_data.drop(labels='size_category', axis=1)
X


# In[14]:


y=forest_data[['size_category']]
y


# In[15]:


#Normalising the data as there is scale difference
from sklearn.preprocessing import StandardScaler
scaler   = StandardScaler()
scaled_X = scaler.fit_transform(X)


# In[19]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(scaled_X,y,test_size=0.20,random_state= 12,stratify=y)
X_train.shape,y_train.shape


# In[18]:


X_test.shape,y_test.shape


# In[59]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
encoded_y=le.fit_transform(y)
encoded_y


# ## Kernel Linear 
# 
# ### Model Training  | Model Testing | Model Evaluation
# 
# 

# In[66]:


from sklearn import svm 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score

import warnings 
warnings.filterwarnings('ignore')

X_train,X_test,y_train,y_test = train_test_split(scaled_X,encoded_y,test_size = 0.20,random_state = 12,stratify = encoded_y)

svc_classifier = SVC(kernel='linear')
svc_classifier.fit(X_train,y_train)
y_pred = svc_classifier.predict(X_test)



print("Overall Accuracy : ",accuracy_score(y_test,y_pred))
print("AUC Score        : ",roc_auc_score(y_test,y_pred))
print("AccuracyScore    : ",accuracy_score(y_test,y_pred))


print("Recall           : ",recall_score(y_test,y_pred))
print("Precision        : ",precision_score(y_test,y_pred))

print("Confusion Matrix :")

plt.figure(figsize = (8,6))

confu_matrix = confusion_matrix(y_test,y_pred)
sns.heatmap(data = confu_matrix,
            annot=True,
            linewidths=0.8)
plt.xlabel("Prediction")
plt.ylabel("Actual labels")
plt.title("Prediction Vs Actual",size = 20)


# ### Kernel - Rbf

# In[69]:


from sklearn import svm 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score

import warnings 
warnings.filterwarnings('ignore')

X_train,X_test,y_train,y_test = train_test_split(scaled_X,encoded_y,test_size = 0.20,random_state = 12,stratify = encoded_y)

svc_classifier = SVC(kernel='linear')
svc_classifier.fit(X_train,y_train)
y_pred = svc_classifier.predict(X_test)



print("Overall Accuracy : ",accuracy_score(y_test,y_pred))
print("AUC Score        : ",roc_auc_score(y_test,y_pred))
print("AccuracyScore    : ",accuracy_score(y_test,y_pred))


print("Recall           : ",recall_score(y_test,y_pred))
print("Precision        : ",precision_score(y_test,y_pred))

print("Confusion Matrix :")

plt.figure(figsize = (8,6))

confu_matrix = confusion_matrix(y_test,y_pred)
sns.heatmap(data = confu_matrix,
            annot=True,
            linewidths=0.8)
plt.xlabel("Prediction")
plt.ylabel("Actual labels")
plt.title("Prediction Vs Actual",size = 20)


# In[ ]:




