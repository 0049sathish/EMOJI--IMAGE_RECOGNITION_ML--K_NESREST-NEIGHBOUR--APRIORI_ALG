#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


pip install mlxtend


# In[4]:


from mlxtend.frequent_patterns import apriori,association_rules


# In[5]:


data = pd.read_csv('retail_dataset.csv')


# In[6]:


data


# In[7]:


#unique datas


# In[8]:


items = (data['0'].unique())
items


# In[9]:


#data process


# In[10]:


itemset = set(items)
encoded_values =[]
for index,row in data.iterrows():
    rowset = set(row)
    labels ={}
    uncommons = list(itemset - rowset)
    commons = list(itemset.intersection(rowset))
    for uc in uncommons:
        labels[uc] = 0
    for com in commons:
        labels[com] = 1
    encoded_values.append(labels)
encoded_values[0]


# In[11]:


# data as dataframe


# In[16]:


binary_data = pd.DataFrame(encoded_values)
binary_data


# In[17]:


#Applying apriori


# In[19]:


frequent_items = apriori(binary_data, min_support=0.2, use_colnames=True, verbose=1)


# In[21]:


frequent_items.head(10)


# In[22]:


#association


# In[23]:


rules = association_rules(frequent_items,metric='confidence',min_threshold=0.6)


# In[24]:


rules


# In[25]:


#rules output


# In[28]:


for i in range(14):
    print("rule:",rules.antecedents[i],"-->",rules.consequents[i])
    print("support:",rules.support[i])
    print("confidence:",rules.confidence[i])
    print("****************************************************")


# In[29]:


#visualization


# In[31]:


plt.scatter(rules['support'],rules['confidence'],alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('support vs confidence')
plt.show()


# In[32]:


#support vs lift


# In[33]:


plt.scatter(rules['support'],rules['lift'],alpha=0.5)
plt.xlabel('support')
plt.ylabel('lift')
plt.title('support vs lift')
plt.show()


# In[34]:


#lift vs confidence


# In[38]:


fit = np.polyfit(rules['lift'],rules['confidence'],1)
fit_fn = np.poly1d(fit)
plt.plot(rules['lift'],rules['confidence'],'yo',rules['lift'],
fit_fn(rules['lift']))


# In[ ]:




