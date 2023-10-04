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


cool=os.listdir("C:/Users/aedpu/OneDrive/Documents/s.k ml/EMOJI/cool")


# In[11]:


angry=os.listdir("C:/Users/aedpu/OneDrive/Documents/s.k ml/EMOJI/angry")


# In[12]:


shy=os.listdir("C:/Users/aedpu/OneDrive/Documents/s.k ml/EMOJI/shy")


# In[13]:


limit=10
cool_image=[None]*limit
j=0
for i in cool:
    if(j<limit):
        cool_image[j]=imread("C:/Users/aedpu/OneDrive/Documents/s.k ml/EMOJI/cool/"+i)
        j+=1
    else:
        break     


# In[15]:


cool_image


# In[16]:


limit=10
angry_image=[None]*limit
j=0
for i in angry:
    if(j<limit):
        angry_image[j]=imread("C:/Users/aedpu/OneDrive/Documents/s.k ml/EMOJI/angry/"+i)
        j+=1
    else:
        break


# In[17]:


limit=10
shy_image=[None]*limit
j=0
for i in shy:
    if(j<limit):
        shy_image[j]=imread("C:/Users/aedpu/OneDrive/Documents/s.k ml/EMOJI/shy/"+i)
        j+=1
    else:
        break


# In[18]:


imshow(cool_image[2])


# In[19]:


imshow(angry_image[8])


# In[21]:


imshow(shy_image[3])


# In[23]:


cool_gray=[None]*limit
j=0
for i in cool:
    if(j<limit):
        cool_gray[j]=rgb2gray(cool_image[j])
        j+=1
    else:
        break


# In[24]:


angry_gray=[None]*limit
j=0
for i in angry:
    if(j<limit):
        angry_gray[j]=rgb2gray(angry_image[j])
        j+=1
    else:
        break


# In[25]:


shy_gray=[None]*limit
j=0
for i in shy:
    if(j<limit):
        shy_gray[j]=rgb2gray(shy_image[j])
        j+=1
    else:
        break


# In[26]:


cool_gray[2].shape


# In[27]:


angry_gray[8].shape


# In[28]:


shy_gray[6].shape


# In[29]:


imshow(cool_gray[2])


# In[30]:


imshow(angry_gray[2])


# In[31]:


imshow(shy_gray[5])


# In[32]:


for j in range(10):
    cool=cool_gray[j]
    cool_gray[j]=resize(cool,(512,512))


# In[33]:


for j in range (10):
    angry=angry_gray[j]
    angry_gray[j]=resize(angry,(512,512))


# In[34]:


for j in range (10):
    shy=shy_gray[j]
    shy_gray[j]=resize(shy,(512,512))


# In[35]:


## cool


# In[38]:


len_of_images_cool=len(cool_gray)


# In[39]:


len_of_images_cool


# In[40]:


image_size_cool=cool_gray[1].shape


# In[41]:


image_size_cool


# In[42]:


flatten_size_cool=image_size_cool[0]*image_size_cool[1]


# In[43]:


flatten_size_cool


# In[44]:


for i in range(len_of_images_cool):
    cool_gray[i]=np.ndarray.flatten(cool_gray[i]).reshape(flatten_size_cool,1)


# In[45]:


cool_gray=np.dstack(cool_gray)


# In[46]:


cool_gray


# In[47]:


cool_gray.shape


# In[48]:


cool_gray=np.rollaxis(cool_gray,axis=2,start=0)


# In[49]:


cool_gray.shape


# In[50]:


cool_gray=cool_gray.reshape(len_of_images_cool,flatten_size_cool)


# In[51]:


cool_gray.shape


# In[52]:


cool_data=pd.DataFrame(cool_gray)


# In[53]:


cool_data["lable"]="cool"


# In[54]:


cool_data


# In[57]:


## angry


# In[58]:


len_of_images_angry=len(angry_gray)


# In[59]:


len_of_images_angry


# In[60]:


image_size_angry=angry_gray[1].shape


# In[61]:


image_size_angry


# In[62]:


flatten_size_angry=image_size_angry[0]*image_size_angry[1]


# In[63]:


flatten_size_angry


# In[64]:


for i in range(len_of_images_angry):
    angry_gray[i]=np.ndarray.flatten(angry_gray[i]).reshape(flatten_size_angry,1)


# In[65]:


angry_gray=np.dstack(angry_gray)


# In[66]:


angry_gray


# In[67]:


angry_gray.shape


# In[68]:


angry_gray=np.rollaxis(angry_gray,axis=2,start=0)


# In[69]:


angry_gray.shape


# In[70]:


angry_gray=angry_gray.reshape(len_of_images_angry,flatten_size_angry)


# In[71]:


angry_gray.shape


# In[72]:


angry_data=pd.DataFrame(angry_gray)


# In[73]:


angry_data["lable"]="angry"


# In[74]:


angry_data


# In[77]:


#shy


# In[78]:


len_of_images_shy=len(shy_gray)


# In[79]:


len_of_images_shy


# In[80]:


image_size_shy=shy_gray[1].shape


# In[81]:


image_size_shy


# In[82]:


flatten_size_shy=image_size_shy[0]*image_size_shy[1]


# In[83]:


flatten_size_shy


# In[84]:


for i in range(len_of_images_shy):
    shy_gray[i]=np.ndarray.flatten(shy_gray[i]).reshape(flatten_size_shy,1)


# In[85]:


shy_gray=np.dstack(shy_gray)


# In[86]:


shy_gray


# In[87]:


shy_gray.shape


# In[88]:


shy_gray=np.rollaxis(shy_gray,axis=2,start=0)


# In[89]:


shy_gray.shape


# In[90]:


shy_gray=shy_gray.reshape(len_of_images_shy,flatten_size_shy)


# In[91]:


shy_gray.shape


# In[92]:


shy_data=pd.DataFrame(shy_gray)


# In[93]:


shy_data["lable"]="shy"


# In[94]:


shy_data


# In[529]:


emoji_1=pd.concat([cool_data,angry_data])


# In[530]:


emoji=pd.concat([emoji_1,shy_data])


# In[531]:


emoji


# In[532]:


from sklearn.utils import shuffle


# In[533]:


felling_indexed=shuffle(emoji).reset_index()


# In[534]:


felling_indexed


# In[535]:


felling_emoji=felling_indexed.drop(['index'],axis=1)


# In[536]:


felling_emoji


# In[537]:


felling_emoji.to_csv("emoji.csv")


# In[538]:


y=felling_emoji.values[:,-1]


# In[539]:


from sklearn.model_selection import train_test_split


# In[540]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[541]:


x=felling_emoji.values[:,:-1]


# In[542]:


x_train.shape


# In[543]:


x_test.shape


# In[544]:


from sklearn import decomposition


# In[545]:


pca = decomposition.PCA(n_components=20,whiten=True,random_state=1)


# In[546]:


pca.fit(x_train)


# In[547]:


x_train_pca=pca.transform(x_train)


# In[548]:


x_test_pca=pca.transform(x_test)


# In[549]:


x_train_pca.shape


# In[550]:


x_test_pca.shape


# In[551]:


eigen =(np.reshape(x[10],(512,512)).astype(np.float64))


# In[552]:


eigen


# In[553]:


#plotting images


# In[554]:


fig = plt.figure(figsize=(30,30))
for i in range(10):
    ax = fig.add_subplot(2,5,i+1,xticks=[],yticks=[])
    ax.imshow(pca.components_[i].reshape(eigen.shape),cmap=plt.cm.bone)


# In[555]:


from sklearn import svm


# In[556]:


clf = svm.SVC(C=2,gamma=0.006,kernel='rbf')
clf.fit(x_train_pca,y_train)


# In[557]:


y_pred = clf.predict(x_test_pca)
y_pred


# In[558]:


for i in (np.random.randint(0,6,6)):
    predicted_images =(np.reshape(x_test[i],(512,512)).astype(np.float64))
    plt.title('predicted lable:{0}'.format(y_pred[i]))
    plt.imshow(predicted_images,interpolation='nearest',cmap='gray')
    plt.show()


# In[559]:


from sklearn import metrics


# In[560]:


accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy


# In[ ]:





# In[ ]:




