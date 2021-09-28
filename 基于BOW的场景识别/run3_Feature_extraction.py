#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import os
from scipy.cluster.vq import *
from sklearn import preprocessing
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import joblib
from sklearn.cluster import MiniBatchKMeans

from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn import svm


# In[2]:


def sift(path, features):
    data  = []
    label = []
    des_list = []
    for i,category in enumerate(features): 
        for filename in os.listdir(path+category):
            if filename == '.DS_Store':
                continue
            img = cv2.imread(path+category+'/'+filename, 0)
            sift = cv2.xfeatures2d.SIFT_create()
            #Calculate the feature point and feature point description of the picture
            keypoints, f = sift.detectAndCompute(img, None)
            des_list.append(f)
            label.append(i)
    return np.array(des_list),label


# In[3]:


def dense_sift(path, features):
    data  = []
    des_list = []
    for i,category in enumerate(features): 
        for filename in os.listdir(path+category):
            if filename == '.DS_Store':
                continue
            img = cv2.imread(path+category+'/'+filename, 0)
            sift = cv2.xfeatures2d.SIFT_create()
            #Calculate the feature point and feature point description of the picture
            r, c = img.shape
            keypoint= []
            #Divide the picture into 8×8 blocks
            for lrow in range(8, r-8 , 8):
                for lcol in range(8, c-8 , 8):
                    kp = cv2.KeyPoint(lcol, lrow, 8, _class_id=0)
                    keypoint.append(kp)
            keypoints, features = sift.compute(img, keypoint)
            des_list.append(features)
            
    return np.array(des_list)


# In[4]:


def bow(des_list): 
    Minikmeans = MiniBatchKMeans(n_clusters=500,batch_size=1000)
    print("start")
    Minikmeans.fit(des_list)
    #joblib.dump(Minikmeans,'run3_kmeans500_1000.pkl')
    word = Minikmeans.cluster_centers_
    labels = Minikmeans.labels_
    return word,labels,Minikmeans


# In[5]:


def count_word(des_list,labels):
    des_num=[]
    for n in des_list:
        des_num.append(n.shape[0])
    des_num=np.array(des_num)    
    img_words=np.zeros([len(des_num),500])
    z=0
    a=0
    for n in des_num:
        for i in range(0,n):
            w=labels[a]
            img_words[z][w]+= 1
            a=a+1
        z+=1
    return img_words


# In[6]:


if __name__ == '__main__':
    #训练集路径
    train_path = 'training/'
    #测试集路径
    test_path  = 'testing/'
    #数据集类别数
    features = os.listdir(train_path)
    features.remove('.DS_Store')
    #print(features)
    #提取图像和标签
    s_list,label= sift(train_path, features)
    #np.save("des_list.npy",s_list)
    descriptors = s_list[0]
    for descriptor in s_list[1:]:
        descriptors = np.vstack((descriptors, descriptor))
    
    word,labels,model=bow(descriptors)
    sift_train=count_word(s_list,labels) 
    #np.save("features_SIFT500.npy",sift_train)
    
    
    #denseSIFT
    d_list= dense_sift(train_path, features)
    #np.save("des_list.npy",des_list)
    descriptors = d_list[0]
    for descriptor in d_list[1:]:
        descriptors = np.vstack((descriptors, descriptor))
    
    word_d,labels_d,model_d=bow(descriptors)
    dense_train=count_word(d_list,labels_d)
    #np.save("features_denseSIFT53_1000.npy",dense_train)  


# In[10]:


#classifer

y = np.array(label)
X_sift = sift_train
X_dense = dense_train

scaler = StandardScaler()


X1_train, X1_test, y1_train, y1_test = train_test_split(X_sift, y, test_size=0.3, random_state=3, stratify=y)
X2_train, X2_test, y2_train, y2_test = train_test_split(X_dense, y, test_size=0.3, random_state=3, stratify=y)



# k-fold cross validation
kf = StratifiedKFold(shuffle=True, random_state=None)

gaussian_np = np.zeros((0, 1))
svm_ploy = np.zeros((0, 1))
svm_rbf = np.zeros((0, 1))

i = 0
for train_index, test_index in kf.split(X_sift, y):
    X_train, X_test =X_sift[train_index], X_sift[test_index]
    y_train, y_test = y[train_index], y[test_index]

    #print("The", i, "times:" + 'SIFT')
    # Create a Bayesian classifier object
    gnb = GaussianNB()
    # populate the data
    gauss_clf = gnb.fit(X_train, y_train)
    y_pred = gauss_clf.predict(X_test)

    gaussian_accuracy = (y_test == y_pred).sum() / X_test.shape[0]
    print("GaussianNB:Accuracy is : %2f" % gaussian_accuracy)
    gaussian_np = np.append(gaussian_np, gaussian_accuracy)


    # create a SVM classifier whose kernel function is rbf
    regr = svm.SVC(kernel='rbf')
    # populate the data
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    print("rbf SVM:Accuracy is : %2f" % ((y_test == y_pred).sum() / X_test.shape[0]))
    svm_accuracy = (y_test == y_pred).sum() / X_test.shape[0]
    np.append(svm_ploy, svm_accuracy)
    print("--------")
    i = i + 1

print('SIFT Gaussian avg: %f' % gaussian_np.mean())
print('SIFT svm avg: %f' % svm_accuracy.mean())

for train_index, test_index in kf.split(X_dense, y):
    X_train, X_test = X_dense[train_index], X_dense[test_index]
    y_train, y_test = y[train_index], y[test_index]

    #print("The", i, "times:" +'dense_SIFT')
    # Create a Bayesian classifier object
    gnb = GaussianNB()
    # populate the data
    gauss_clf = gnb.fit(X_train, y_train)
    y_pred = gauss_clf.predict(X_test)

    gaussian_accuracy = (y_test == y_pred).sum() / X_test.shape[0]
    print("GaussianNB:Accuracy is : %2f" % gaussian_accuracy)
    gaussian_np = np.append(gaussian_np, gaussian_accuracy)


    # create a SVM classifier whose kernel function is rbf
    regr = svm.SVC(kernel='rbf')
    # populate the data
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    print("rbf SVM:Accuracy is : %2f" % ((y_test == y_pred).sum() / X_test.shape[0]))
    svm_accuracy = (y_test == y_pred).sum() / X_test.shape[0]
    np.append(svm_ploy, svm_accuracy)
    print("--------")
    i = i + 1

print('dense_SIFT Gaussian avg: %f' % gaussian_np.mean())
print('dense_SIFT svm avg: %f' % svm_accuracy.mean())




# In[ ]:




