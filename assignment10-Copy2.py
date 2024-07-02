#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

np.e


# In[ ]:





# In[2]:


from sklearn.datasets import load_wine
wine = load_wine()
wine.keys()
X = wine['data']
y = wine['target']

print(X.shape)
print(y.shape)


# In[3]:


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# In[4]:


from sklearn.linear_model import LogisticRegression


model = LogisticRegression()
model.fit(X_train,y_train)


# In[5]:


y_pred = model.predict(X_test)
y_pred


# In[6]:


from sklearn.metrics import accuracy_score,\
confusion_matrix,\
classification_report

cm = confusion_matrix(y_test,y_pred)


# In[7]:


from sklearn.metrics import accuracy_score,\
confusion_matrix,\
classification_report

cm = confusion_matrix(y_test,y_pred)
cm


# In[8]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(cm,annot=True)
plt.show()


# In[9]:


accuracy = accuracy_score(y_test,y_pred)
accuracy


# In[10]:


cr = classification_report(y_test,y_pred)
print(cr)


# In[11]:


from sklearn.datasets import load_wine
    


# In[12]:


data = load_wine()

print(data['DESCR'])


# In[13]:


import numpy as np

np.e


# In[14]:


import pandas as pd 
path=(r"C:\Users\Student\Downloads\iris.data.csv")

data=pd.read_csv(path)

data.head()


# In[15]:


from sklearn.datasets import load_iris
flower = load_iris()
flower.keys()
X = flower['data']
y = flower['target']

print(X.shape)
print(y.shape)


# In[16]:


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# In[17]:


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train,y_train)


# In[18]:


y_pred = model.predict(X_test)
y_pred


# In[30]:


import pandas as pd 

df=pd.DataFrame(X,
               columns=flower['feature_names'])

df['target'] = y
df.sample()


# In[29]:


flower['target_names']


# In[21]:


from sklearn.metrics import accuracy_score,\
confusion_matrix,\
classification_report

cm = confusion_matrix(y_test,y_pred)
cm


# In[22]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(cm,annot=True)
plt.show()


# In[23]:


accuracy = accuracy_score(y_test,y_pred)
accuracy


# In[24]:


cr = classification_report(y_test,y_pred)
print(cr)


# In[ ]:




