#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
This code includes: 
 the training set processing 
 the establishment of the bag of words 
 the calculation of the image feature histogram
 SVM classifier
 the test set processing
 prediction
 

Generate file:
    simples_process.npy              #Training set sampling results
    simples_list.npy                 #List of the number of samples in the training set
    training_truelabel.npy           #True label of training set image
    train_features_sort.npy          #Train_set image feature histogram
    
    test_simples_process_sort.npy    #test set sampling results
    test_simples_list_sort.npy       #List of the number of samples in the test set
    test_features_sort.npy           #test_set image feature histogram
    
    run2.txt 
'''


# In[31]:


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
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[32]:


def read_img(path, features):
    data  = []
    label = []
    label_word = []
    for i,category in enumerate(features):
        for filename in os.listdir(path+category):
            if filename == '.DS_Store':
                continue
            img = cv2.imread(path+category+'/'+filename, 0)
            data.append(img)
            label_word.append(category)
            label.append(i)
    return data, label,label_word


# In[33]:


def read_test_img(path):
    data  = []
    a=0
    image_names = os.listdir(path)
    image_names.sort(key=lambda x:int(x[:-4]))
    for filename in image_names:
        img = cv2.imread(path+'/'+filename, 0)
        data.append(img)
    return data,image_names


# In[34]:


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


# In[35]:


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
        print(i,p.shape)
        patch_stack= np.vstack((patch_stack,p))
        patch_list.append(p.shape[0])
    return np.array(patch_stack),np.array(patch_list)


# In[36]:


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
    return word,labels,Minikmeans


# In[37]:


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


# In[43]:


def norm(train_X, test_X):
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.transform(test_X)
    return train_X, test_X


# In[44]:


def classifier(train, trainlabel):
    '''
    SVM linear classifier
    '''
    '''
    Call the function to evaluate in the set parameter range and 
    select the optimal solution.
    
    '''
    '''
    
    
    from sklearn.model_selection import GridSearchCV

    svc_model = svm.SVC(kernel='linear', decision_function_shape='ovo')

    # use cross-validation to choose the parameters
    param_grid = {'C': [0.012, 0.013, 0.011],
              'gamma': [0.0005, 0.0006, 0.0004]
              }

    grid_search = GridSearchCV(svc_model, param_grid, n_jobs=-1, verbose=1)
    grid_search.fit(train_X, train_y.ravel())

    # print out the best parameters
    print("best parameters are" % grid_search.best_params_, grid_search.best_params_)
    
    
    '''
    n_class = 15
    x = train[:100*n_class]
    y = trainlabel[:100*n_class]

    train_X, test_X, train_y, test_y = train_test_split(x, y, train_size=.9, random_state=66, stratify=y)
    train_X, test_X = norm(train_X, test_X)

    clf = svm.SVC(kernel='linear', C=0.01, gamma=0.0005, decision_function_shape='ovo')
    clf.fit(train_X,train_y)

    out = clf.predict(train_X)
    print("train: ",acc(out, train_y))
    out = clf.predict(test_X)
    print("test : ",acc(out, test_y))
    return clf


# In[40]:


def acc(y,yhat):
    return (len(y)-len(np.nonzero(y-yhat)[0]))/len(y)*100


# In[61]:


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
    train_data, trainlabel,label_word = read_img(train_path, features)
    #np.save("training_truelabel.npy",trainlabel)
    
    #Get all samples
    patch_stack,patch_list=all_patch(train_data)
    patch_stack=patch_stack[1:,:]
    
    #data processing
    process_patch= preprocessing.scale(patch_stack, axis=1)
    
    #Save sampling results
    #np.save("simples_process.npy",process_patch)
    #Save the number of samples corresponding to each image
    #np.save("simples_list.npy",patch_list)
    
    #Download result
    #patch_stack=np.load("simples_process.npy")
    #patch_list=np.load("simples_list.npy")
    
    #Build a bag of words
    word,labels,Minikmeans=kmeans_vocabulary(patch_stack)
    
    #Download kmeans model
    #model_kmeans = joblib.load("minibatchkmeans_model.pkl")

    
    #Count each picture, 500 vectors correspond to 500 words, count label
    img_words=count_word(patch_list,labels)
    #np.save("train_features.npy",img_words)
    train=img_words
    
    
    #Processing test set data
    #Extract test image
    test_data,image_names= read_test_img(test_path)
    
    test_patch_stack,test_patch_list=all_patch(test_data)
    test_patch_stack=test_patch_stack[1:,:]

    #data processing
    test_process_patch= preprocessing.scale(test_patch_stack, axis=1)

    #Save sampling results
    #np.save("test_simples_process_sort.npy",test_process_patch)
    #np.save("test_simples_list_sort.npy",test_patch_list)
    #test_process_patch=np.load("test_simples_process_sort.npy")
    #test_patch_list=np.load("test_simples_list_sort.npy")
    
    #Generate image feature histogram
    test_labels=Minikmeans.predict(test_process_patch)

    test=count_word(test_patch_list,test_labels)
    #np.save("test_features.npy",test)
    
    #StandardScaler
    scaler = StandardScaler()
    train1 = scaler.fit_transform(train)
    test1 = scaler.transform(test)
    
    clf = classifier(train, trainlabel)
    prediction = clf.predict(test1)
    #np.save("test_features_sort.npy",prediction)
    #prediction= np.load('test_pre_sort.npy')
    #lclass=['Forest','bedroom','Office','Highway','Coast',
    #    'Insidecity','TallBuilding','industrial','Street',
    #    'livingroom','Suburb','Mountain','kitchen','OpenCountry','store']
    
    predicted_class=[]
    
    for i in prediction:
        predicted_class.append(features[i])
    file = open('run2.txt', 'w')
    for i in zip(image_names, predicted_class):
        file.write(i[0]+" "+i[1]+'\n')
    file.close()
    


# In[ ]:




