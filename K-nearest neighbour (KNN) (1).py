#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.simplefilter("ignore")


# In[2]:


import numpy as np


# In[3]:


#pip uninstall pandas


# In[4]:


import pandas as pd


# In[5]:


pd.__version__


# In[6]:


import matplotlib.pyplot as plt


# In[7]:


pip install matplotlib


# In[8]:


import matplotlib.pyplot as plt


# In[9]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:


pip install seaborn


# In[11]:


import seaborn as sns


# In[12]:


pip install openpyxl


# In[13]:


dataset = pd.read_excel('student_dataset.xlsx')


# In[14]:


dataset


# In[15]:


dataset.shape


# In[16]:


dataset.head()


# In[17]:


dataset.tail()


# In[18]:


data=dataset.drop(['S.NO','TOTAL'],axis=1)


# In[19]:


data


# In[20]:


data['certificate'].unique()


# In[21]:


dataset.groupby('certificate').size()


# In[22]:


pip install scikit-learn


# In[23]:


import sklearn


# In[24]:


from sklearn.preprocessing import LabelEncoder


# In[25]:


data.iloc[:,3] = LabelEncoder().fit_transform(data.iloc[:,3])


# In[26]:


data


# In[27]:


x = data.iloc[:,:-1].values
x


# In[28]:


type(x)


# In[29]:


y = data.iloc[:,3]
y


# In[30]:


x_frame = pd.DataFrame(x,columns=['Attn','Theory','daily assn'])
x_frame


# In[31]:


y_frame = pd.DataFrame(y)
y_frame


# In[32]:


type(y_frame)


# In[33]:


from sklearn.model_selection import train_test_split


# In[34]:


x_train,x_test,y_train,y_test = train_test_split(x_frame,y_frame,test_size=0.2,random_state=0)


# In[35]:


x_test


# In[36]:


type(x_test)


# In[37]:


y_train


# In[38]:


x_test


# In[39]:


type(y_test)


# In[40]:


y_test


# In[41]:


from sklearn.neighbors import KNeighborsClassifier


# In[42]:


knn = KNeighborsClassifier(n_neighbors=5)


# In[43]:


knn.fit(x_train,y_train)


# In[44]:


y_predict = knn.predict(x_test)
y_predict


# In[45]:


knn.score(x_train,y_train)


# In[46]:


knn.score(x_test,y_test)


# In[51]:


pip install sklearn3


# In[53]:


from sklearn.metrics import confusion_matrix


# In[54]:


con_matrix = confusion_matrix(y_test,y_predict)
con_matrix


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




