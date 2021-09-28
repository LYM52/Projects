# -*-coding:utf-8-*-
from sklearn.neighbors import KNeighborsClassifier
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn import preprocessing
from PIL import Image
import os

'''
crop image
currdir: output path
path: the path of image
filename: the name of image
'''
def crop(currdir: str, path: str, filename: str):
    full = os.path.join(path, filename)
    #open image
    im = Image.open(full)
    #Get the width and height, find the minimum value
    width = im.size[0]
    height = im.size[1]
    minlen = width
    if height < minlen:
        minlen = height
    w, h = minlen, minlen
    center_x = im.size[0] / 2
    center_y = im.size[1] / 2
    #crop image
    im = im.crop((center_x - w / 2, center_y - h / 2, center_x + w / 2, center_y + h / 2))
    resize_size = 16
    #zoom image
    region = im.resize((resize_size, resize_size))
    arr = path.split('/')
    resdir = os.path.join(currdir, arr[len(arr) - 1])
    if not os.path.exists(resdir):
        os.mkdir(resdir)
    #save image to output path
    region.save(os.path.join(resdir, filename))

def generate_train_feature():
    g = os.walk("./training/")
    outputdir = os.path.join(os.getcwd(), "train_feature")

    for path, dir_list, file_list in g:
        for file_name in file_list:
            if not file_name.endswith('jpg'):
                continue
            crop(outputdir, path, file_name)



def crop_test(currdir: str, path: str, filename: str):
    full = os.path.join(path, filename)
    im = Image.open(full)
    width = im.size[0]
    height = im.size[1]
    minlen = width
    if height < minlen:
        minlen = height
    w, h = minlen, minlen
    center_x = im.size[0] / 2
    center_y = im.size[1] / 2
    im = im.crop((center_x - w / 2, center_y - h / 2, center_x + w / 2, center_y + h / 2))
    resize_size = 16
    region = im.resize((resize_size, resize_size))
    arr = path.split('/')
    resdir = os.path.join(currdir, arr[len(arr) - 1])
    if not os.path.exists(resdir):
        os.mkdir(resdir)
    region.save(os.path.join(resdir, filename))


def generate_test_feature():
    g_test = os.walk("./testing/")
    outputdir = os.path.join(os.getcwd(), "testing_feature")

    for path, dir_list, file_list in g_test:
        for file_name in file_list:
            if not file_name.endswith('jpg'):
                continue
            crop_test(outputdir, path, file_name)


'''
calculate the accuracy
'''
def correct_percentage(t, t_hat):
    correct = 0.0
    total = len(t) + 1

    for i in range(len(t)):
        if t[i] == t_hat[i]:
            correct = correct + 1
    return (100.0 * correct) / total


'''
convert image to vector
'''
def img2vec(path: str):
    raw_image = Image.open(path)
    image_array = np.array(raw_image)
    w, h = image_array.shape[0], image_array.shape[1]
    line = np.reshape(image_array, (1, w * h))
    return line

feature_dir = 'train_feature'
test_feature_dir = 'testing_feature'

#Dictionary: features > numbers
f2v = dict()
#Dictionary: numbers > features
v2f = dict()

if __name__ == '__main__':
    #generating training data features
    generate_train_feature()
    #generating testing data features
    generate_test_feature()

    #initializing the dictionary of number and feature transformation
    features = os.listdir(feature_dir)
    features_count = len(features)
    for i in range(len(features)):
        item = features[i]
        f2v[item] = i
        v2f[i] = item


    #initialize the training set and convert the image to matrix
    train_x = np.zeros((0, 256))
    train_y = np.zeros((0, 1))

    for i in features:
        current = os.path.join(feature_dir, i)
        image_list = os.listdir(current)
        for j in image_list:
            current_img_path = os.path.join(feature_dir, i, j)
            line = img2vec(current_img_path)
            train_x = np.append(train_x, line, axis=0)
            train_y = np.append(train_y, f2v[i])

    #initialize the testing set and convert the image to matrix
    test_x = np.zeros((0, 256))
    test_x_name = list()
    test_list = os.listdir(test_feature_dir)
    test_list.sort(key=lambda x: int(x[:-4]))
    for i in test_list:
        current_img_path = os.path.join(test_feature_dir, i)
        line = img2vec(current_img_path)
        test_x = np.append(test_x, line, axis=0)
        test_x_name.append(i)



    for i in range(10):
        #for the current K value, 10 times cross-validation was performed
        kf = KFold(10, shuffle=True)
        loopNum = 0
        current_correct = 0
        for train_index, test_index in kf.split(train_x):
            X_train, X_test = train_x[train_index], train_x[test_index]
            y_train, y_test = train_y[train_index], train_y[test_index]

            neigh = KNeighborsClassifier(n_neighbors=i + 1)
            neigh.fit(X_train, y_train)

            y_predict = neigh.predict(X_test)
            current_correct += correct_percentage(y_predict, y_test)
        print("current neighbors = ", i + 1)
        print(current_correct / 10)

    #initialize a KNN classifier with K value of 5
    neigh = KNeighborsClassifier(n_neighbors=5)
    #populate with data
    neigh.fit(train_x, train_y)
    pred_vec = neigh.predict(test_x)

    #save the test set results to run1.txt
    result = ""
    for i in range(len(test_x)):
        vec_idx = pred_vec[i]
        name = v2f[vec_idx]
        result = result + test_list[i] + ' ' + name + '\n'
    with open("run1.txt", 'w') as f:
        f.write(result)

    print("ok")
    #verify the number of test set result classifications
    print(len(np.unique(pred_vec)))

