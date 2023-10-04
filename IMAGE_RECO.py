#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os


# In[2]:


import warnings


# In[3]:


warnings.simplefilter('ignore')


# In[4]:


import numpy as np
import pandas as pd    


# In[5]:


np.version.version


# In[6]:


os.system('pip install numpy==1.24.2')


# In[7]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


pip install scikit-image


# In[9]:


from skimage.io import imread,imshow
from skimage.transform import resize
from skimage.color import rgb2gray


# In[10]:


ak=os.listdir("C:/Users/aedpu/OneDrive/Documents/s.k ml/ACTORS/AK")


# In[11]:


joneydepp=os.listdir("C:/Users/aedpu/OneDrive/Documents/s.k ml/ACTORS/joney depp")


# In[12]:


rashmi=os.listdir("C:/Users/aedpu/OneDrive/Documents/s.k ml/ACTORS/Rashmi")


# In[13]:


limit=10
ak_image=[None]*limit
j=0
for i in ak:
    if(j<limit):
        ak_image[j]=imread("C:/Users/aedpu/OneDrive/Documents/s.k ml/ACTORS/AK/"+i)
        j+=1
    else:
        break     


# In[14]:


ak_image


# In[15]:


limit=10
joneydepp_image=[None]*limit
j=0
for i in joneydepp:
    if(j<limit):
        joneydepp_image[j]=imread("C:/Users/aedpu/OneDrive/Documents/s.k ml/ACTORS/joney depp/"+i)
        j+=1
    else:
        break


# In[16]:


limit=10
rashmi_image=[None]*limit
j=0
for i in rashmi:
    if(j<limit):
        rashmi_image[j]=imread("C:/Users/aedpu/OneDrive/Documents/s.k ml/ACTORS/Rashmi/"+i)
        j+=1
    else:
        break


# In[17]:


imshow(ak_image[2])


# In[18]:


imshow(joneydepp_image[8])


# In[19]:


imshow(rashmi_image[6])


# In[20]:


ak_gray=[None]*limit
j=0
for i in ak:
    if(j<limit):
        ak_gray[j]=rgb2gray(ak_image[j])
        j+=1
    else:
        break


# In[21]:


joneydepp_gray=[None]*limit
j=0
for i in joneydepp:
    if(j<limit):
        joneydepp_gray[j]=rgb2gray(joneydepp_image[j])
        j+=1
    else:
        break


# In[22]:


rashmi_gray=[None]*limit
j=0
for i in rashmi:
    if(j<limit):
        rashmi_gray[j]=rgb2gray(rashmi_image[j])
        j+=1
    else:
        break


# In[23]:


ak_gray[2].shape


# In[24]:


joneydepp_gray[8].shape


# In[25]:


rashmi_gray[6].shape


# In[26]:


imshow(ak_gray[2])


# In[27]:


imshow(joneydepp_gray[2])


# In[28]:


imshow(rashmi_gray[5])


# In[29]:


for j in range(10):
    ak=ak_gray[j]
    ak_gray[j]=resize(ak,(512,512))


# In[30]:


for j in range (10):
    joneydepp=joneydepp_gray[j]
    joneydepp_gray[j]=resize(joneydepp,(512,512))


# In[31]:


for j in range (10):
    rashmi=rashmi_gray[j]
    rashmi_gray[j]=resize(rashmi,(512,512))


# In[32]:


## AK


# In[33]:


len_of_images_ak=len(ak_gray)


# In[34]:


len_of_images_ak


# In[35]:


image_size_ak=ak_gray[1].shape


# In[36]:


image_size_ak


# In[37]:


flatten_size_ak=image_size_ak[0]*image_size_ak[1]


# In[38]:


flatten_size_ak


# In[39]:


for i in range(len_of_images_ak):
    ak_gray[i]=np.ndarray.flatten(ak_gray[i]).reshape(flatten_size_ak,1)


# In[40]:


ak_gray=np.dstack(ak_gray)


# In[41]:


ak_gray


# In[42]:


ak_gray.shape


# In[43]:


ak_gray=np.rollaxis(ak_gray,axis=2,start=0)


# In[44]:


ak_gray.shape


# In[45]:


ak_gray=ak_gray.reshape(len_of_images_ak,flatten_size_ak)


# In[46]:


ak_gray.shape


# In[47]:


ak_data=pd.DataFrame(ak_gray)


# In[48]:


ak_data["lable"]="ak"


# In[49]:


ak_data


# In[50]:


## joneydepp


# In[51]:


len_of_images_joneydepp=len(joneydepp_gray)


# In[52]:


len_of_images_joneydepp


# In[53]:


image_size_joneydepp=joneydepp_gray[1].shape


# In[54]:


image_size_joneydepp


# In[55]:


flatten_size_joneydepp=image_size_joneydepp[0]*image_size_joneydepp[1]


# In[56]:


flatten_size_joneydepp


# In[57]:


for i in range(len_of_images_joneydepp):
    joneydepp_gray[i]=np.ndarray.flatten(joneydepp_gray[i]).reshape(flatten_size_joneydepp,1)


# In[58]:


joneydepp_gray=np.dstack(joneydepp_gray)


# In[59]:


joneydepp_gray


# In[60]:


joneydepp_gray.shape


# In[61]:


joneydepp_gray=np.rollaxis(joneydepp_gray,axis=2,start=0)


# In[62]:


joneydepp_gray.shape


# In[63]:


joneydepp_gray=joneydepp_gray.reshape(len_of_images_joneydepp,flatten_size_joneydepp)


# In[64]:


joneydepp_gray.shape


# In[65]:


joneydepp_data=pd.DataFrame(joneydepp_gray)


# In[66]:


joneydepp_data["lable"]="joneydepp"


# In[67]:


joneydepp_data


# In[68]:


#rashmi


# In[69]:


len_of_images_rashmi=len(rashmi_gray)


# In[70]:


len_of_images_rashmi


# In[71]:


image_size_rashmi=rashmi_gray[1].shape


# In[72]:


image_size_rashmi


# In[73]:


flatten_size_rashmi=image_size_rashmi[0]*image_size_rashmi[1]


# In[74]:


flatten_size_rashmi


# In[75]:


for i in range(len_of_images_rashmi):
    rashmi_gray[i]=np.ndarray.flatten(rashmi_gray[i]).reshape(flatten_size_rashmi,1)


# In[76]:


rashmi_gray=np.dstack(rashmi_gray)


# In[77]:


rashmi_gray


# In[78]:


rashmi_gray.shape


# In[79]:


rashmi_gray=np.rollaxis(rashmi_gray,axis=2,start=0)


# In[80]:


rashmi_gray.shape


# In[81]:


rashmi_gray=rashmi_gray.reshape(len_of_images_rashmi,flatten_size_rashmi)


# In[82]:


rashmi_gray.shape


# In[83]:


rashmi_data=pd.DataFrame(rashmi_gray)


# In[84]:


rashmi_data["lable"]="rashmi"


# In[85]:


rashmi_data


# In[86]:


actor_1=pd.concat([ak_data,joneydepp_data])


# In[91]:


actors=pd.concat([actor_1,rashmi_data])


# In[92]:


actors


# In[93]:


from sklearn.utils import shuffle


# In[94]:


kollywood_indexed=shuffle(actors).reset_index()


# In[95]:


kollywood_indexed


# In[96]:


kollywood_actors=kollywood_indexed.drop(['index'],axis=1)


# In[97]:


kollywood_actors


# In[121]:


kollywood_actors.to_csv("actors.csv")


# In[122]:


x=kollywood_actors.values[:,:-1]


# In[123]:


y=kollywood_actors.values[:,-1]


# In[124]:


from sklearn.model_selection import train_test_split


# In[125]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[126]:


x_train.shape


# In[127]:


x_test.shape


# In[128]:


from sklearn import decomposition


# In[129]:


pca = decomposition.PCA(n_components=20,whiten=True,random_state=1)


# In[130]:


pca.fit(x_train)


# In[131]:


x_train_pca=pca.transform(x_train)


# In[132]:


x_test_pca=pca.transform(x_test)


# In[133]:


x_train_pca.shape


# In[134]:


x_test_pca.shape


# In[135]:


eigen =(np.reshape(x[10],(512,512)).astype(np.float64))


# In[136]:


eigen


# In[137]:


#plotting images


# In[138]:


fig = plt.figure(figsize=(30,30))
for i in range(10):
    ax = fig.add_subplot(2,5,i+1,xticks=[],yticks=[])
    ax.imshow(pca.components_[i].reshape(eigen.shape),cmap=plt.cm.bone)


# In[143]:


from sklearn import svm


# In[144]:


clf = svm.SVC(C=2,gamma=0.006,kernel='rbf')
clf.fit(x_train_pca,y_train)


# In[145]:


y_pred = clf.predict(x_test_pca)
y_pred


# In[146]:


for i in (np.random.randint(0,6,6)):
    predicted_images =(np.reshape(x_test[i],(512,512)).astype(np.float64))
    plt.title('predicted lable:{0}'.format(y_pred[i]))
    plt.imshow(predicted_images,interpolation='nearest',cmap='gray')
    plt.show()


# In[147]:


from sklearn import metrics


# In[148]:


accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy


# In[ ]:





# In[ ]:




