# !/usr/bin/env python
# !-*-coding:utf-8 -*-
# !@Time    :2021/5/3 8:28 PM
# !@Author  :CHAMPLOO
# !@File    :run3.py


from torchbearer import Trial
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import torchvision.models as models
from PIL import Image
import torch
import os

epochs = 60
num_class = 15
batch_size = 32
img_size = 224

# The path of data
train_path = 'drive/MyDrive/Data/scene_data/train'
valid_path = 'drive/MyDrive/Data/scene_data/val'
test_path = 'drive/MyDrive/Data/scene_data/test'
real_test = 'drive/MyDrive/Data/scene_raw/testing'

# data transformers for trainset and testset
train_transform = transforms.Compose([
    transforms.Resize((img_size)),
    transforms.CenterCrop(img_size),
    # data augmentation
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Grayscale(3),
    transforms.Resize((img_size)),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# load the dataset from image folder
train_dataset = ImageFolder(train_path, train_transform)  # size 1215
valid_dataset = ImageFolder(valid_path, test_transform)  # size 150
test_dataset = ImageFolder(test_path, test_transform)  # size 135

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# define the model
class Resnet18(nn.Module):
    def __init__(self, num_class):
        super(Resnet18, self).__init__()
        self.convnet = models.resnet18(pretrained=True)
        self.convnet.fc = nn.Linear(512, num_class)

    def forward(self, X):
        out = self.convnet(X)
        return out


# function to plot the loss-acc graphs
def loss_acc_plot(history):
    train_loss_his = [i['loss'] for i in history]
    valid_loss_his = [i['val_loss'] for i in history]
    train_acc_his = [i['acc'] for i in history]
    valid_acc_his = [i['val_acc'] for i in history]

    epochs = len(train_loss_his)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 7))
    ax[0].plot(range(epochs), train_loss_his, label="train_loss")
    ax[0].plot(range(epochs), valid_loss_his, label="valid_loss")

    ax[1].plot(range(epochs), train_acc_his, label="train_acc")
    ax[1].plot(range(epochs), valid_acc_his, label="valid_acc")

    for i in range(2):
        ax[i].legend()
        ax[i].set_xlabel("epochs", fontsize=16)
        ax[i].set_xlim(0, epochs)
    ax[0].set_title('Loss-plot', fontsize=16)
    ax[1].set_title('Accuracy-plot', fontsize=16)
    plt.show()


# easier version of training using torchbearer
model = Resnet18(num_class).cuda()

loss_function = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), lr=0.00002, weight_decay=0.001)

trial = Trial(model, optimiser, loss_function, metrics=['loss', 'accuracy']).cuda()
trial.with_generators(train_loader, val_generator=valid_loader, test_generator=test_loader)
his = trial.run(epochs=epochs)

# show the loss-acc graph
loss_acc_plot(his)

# save the model
torch.save(model.state_dict(), "./resnet18.weights")

labels = train_dataset.classes

# The classes of dataset
labels = train_dataset.classes


# function to output the prediction txt document
def prediction(model, labels, test_transform):
    predicted_class = []
    model.eval()

    test_path = 'drive/MyDrive/Data/scene_raw/testing/'
    image_names = os.listdir(test_path)
    image_names.sort(key=lambda x: int(x[:-4]))

    for filename in image_names:
        test_img = test_transform(Image.open(test_path + filename)).unsqueeze(0).cuda()
        _, idx = torch.max(model(test_img).data, 1)
        predicted_class.append(labels[idx])

    file = open('run3.txt', 'w')
    for i in zip(image_names, predicted_class):
        file.write(i[0] + " " + i[1] + '\n')
    file.close()

    return


prediction(model, labels, test_transform)



