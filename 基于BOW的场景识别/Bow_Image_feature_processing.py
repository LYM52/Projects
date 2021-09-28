#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''
This code includes: 
 the training set processing 
 the establishment of the bag of words 
 the calculation of the image feature histogram

Generate file:
    training_truelabel.npy  #True label of training set image
    train_features_sort.npy #Train_set image feature histogram
    test_features_sort.npy  #test_set image feature histogram
'''


# In[1]:


import os
import cv2
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import joblib
from sklearn.cluster import MiniBatchKMeans
from sklearn import preprocessing


# In[2]:


def read_img(path, features):
    data  = []
    label = []
    label_word = []
    for i,category in enumerate(features):
        print(i,category)
        for filename in os.listdir(path+category):
            if filename == '.DS_Store':
                continue
            img = cv2.imread(path+category+'/'+filename, 0)
            data.append(img)
            label_word.append(category)
            label.append(i)
            print(category,i)
    return data, label,label_word


# In[ ]:


def read_test_img(path):
    data  = []
    a=0
    image_names = os.listdir(path)
    image_names.sort(key=lambda x:int(x[:-4]))
    for filename in image_names:
        img = cv2.imread(path+'/'+filename, 0)
        data.append(img)
    return data


# In[3]:


def simples_img(img,patch_size=8,stride=4):
    '''
    Patch each image
    
    INPUT:  Image sampling (2Darray)
    OUTPUT: Sampling of each picture (nx64 vectors)
    '''
    patch_list=[]
    hight, width = img.shape
    x = 0
    while (x + patch_size <= hight):
        y = 0
        while (y + patch_size <= width):
            cropped = img[x:x+patch_size, y:y+patch_size]
            cropped = np.concatenate(cropped)
            patch_list.append(cropped)
            y += stride   #滑动步长
        x = x + stride
    return np.array(patch_list)


# In[4]:


def all_patch(data):
    '''
    INPUT:  all pictures 
    OUTPUT: all features,The number of features in each image
    '''
    patch_list=[]
    patch_stack=np.empty([1,64])
    for i, category in enumerate(data):
        #Sampling the image 8×8
        p=simples_img(category)
        #print(i,p.shape)
        patch_stack= np.vstack((patch_stack,p))
        patch_list.append(p.shape[0])
    return np.array(patch_stack),np.array(patch_list)


# In[5]:


def kmeans_vocabulary(patch_stack):
    '''
    all features,The number of features in each image
    Use MiniBatch K-Menas algorithm to cluster feature words
    
    INPUT:  all features
    OUTPUT: Word list in bag of words,Word label of each image
    
    '''
    
    print("Start :%d words, %d key points"%(500, patch_stack.shape[0]))
    #kmeans= KMeans(n_clusters=100,n_init=1)
    Minikmeans = MiniBatchKMeans(n_clusters=500,batch_size=1000)
    Minikmeans.fit(patch_stack)
    #save model
    joblib.dump(Minikmeans,'minibatchkmeans_model.pkl')
    word = Minikmeans.cluster_centers_
    labels = Minikmeans.labels_
    return word,labels


# In[6]:


def count_word(patch_list,labels):
    '''
    Sample a picture
    Count the sampling results in each picture against the bag of words
    
    INPUT:List of sample numbers for each picture, sample classification results
    OUTPUT: (img_num,500)
    '''
    img_patch_num=np.array(patch_list)
    img_words=np.zeros([len(patch_list),500])
    z=0
    a=0
    for n in img_patch_num:
        #print(n)
        for i in range(0,n):
            w=labels[a]
            img_words[z][w]+= 1
            a=a+1
        z+=1
    return img_words


# In[7]:


if __name__ == '__main__':
    
    #Training set path
    train_path = 'training/'
    #Test set path
    test_path  = 'testing/'
    
    #Data set category number
    features = os.listdir(train_path)
    features.remove('.DS_Store')
    features.remove('.ipynb_checkpoints')
        
    #Extract train set images and real labels
    data, label,label_word = read_img(train_path, features)
    np.save("training_truelabel.npy",label)
    
    #Get all samples
    patch_stack,patch_list=all_patch(data)
    patch_stack=patch_stack[1:,:]
    
    #data processing
    process_patch= preprocessing.scale(patch_stack, axis=1)
    
    #Save sampling results
    #np.save("simples.npy",process_patch)
    #Save the number of samples corresponding to each image
    #np.save("simples_list.npy",patch_list)
    
    #Download result
    #patch_stack=np.load("simples.npy")
    #patch_list=np.load("simples_list.npy")
    
    #Build a bag of words
    word,labels=kmeans_vocabulary(patch_stack)
    
    #Download kmeans model
    model_kmeans = joblib.load("minibatchkmeans_model.pkl")
    
    labels = model_kmeans.labels_
    
    #Count each picture, 500 vectors correspond to 500 words, count label
    img_words=count_word(patch_list,labels)
    np.save("train_features.npy",img_words)
    
    
    #Processing test set data
    #Extract test image
    test_data= read_test_img(test_path)
    
    test_patch_stack,test_patch_list=all_patch(test_data)
    test_patch_stack=test_patch_stack[1:,:]

    #data processing
    test_process_patch= preprocessing.scale(test_patch_stack, axis=1)

    #Save sampling results
    #np.save("test_simples_process_sort.npy",test_process_patch)
    #np.save("test_simples_list_sort.npy",test_patch_list)
    
    #Generate image feature histogram
    test_labels=model_kmeans.predict(test_process_patch)

    test_img_words=count_word(test_patch_list,test_labels)
    np.save("test_features_sort.npy",test_img_words)


    

