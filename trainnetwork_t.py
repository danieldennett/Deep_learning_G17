#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 09:15:02 2019

@author: Zhouxin
"""

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from PIL import Image

#%% define parameters
train_path = 'SIGN/sign_mnist_train.csv'  #Path to training csv
N_classes = 26                            #Number of classes

batch = 8               # batch size
ep = 10                 # number of epochs

#%% define dataloader
class SIGN(torch.utils.data.Dataset):
    def __init__(self,csv_file,height,width, transforms=None):
        self.data = pd.read_csv(csv_file)
        self.labels = np.asarray(self.data.iloc[:, 0])
        self.height = height
        self.width = width
        self.transforms = transforms

    def __getitem__(self,index):
        label = self.labels[index]
        img_as_np = np.asarray(self.data.iloc[index][1:]).reshape(28,28).astype('uint8')
        img_as_img = Image.fromarray(img_as_np)
        img_as_img = img_as_img.convert('L')
        if self.transforms is not None:
            img_as_tensor = self.transforms(img_as_img)
        return (img_as_tensor,label)
    def __len__(self):
       return len(self.data.index)
   
#%% load data
transform = transforms.Compose([transforms.ToTensor()])
trainset = SIGN(train_path,28,28,transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch,
                                          shuffle=True, num_workers=1)

#%% define classes
classes = ('A', 'B', 'C', 'D',
           'E', 'F', 'G', 'H',
           'I', 'J', 'K', 'L', 
           'M', 'N', 'O', 'P',
           'Q', 'R', 'S', 'T',
           'U', 'V', 'W', 'X',
           'Y', 'Z')

#%% define neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, 
        #        dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv1 = nn.Conv2d(1, 12, 3, padding=1)
        self.conv2 = nn.Conv2d(12, 24, 3, padding=1)
        # MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, 
        #           return_indices=False, ceil_mode=False)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(24,48, 3, padding=1)
        self.conv4 = nn.Conv2d(48,96, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # Linear(in_features, out_features, bias=True)
        self.fc1 = nn.Linear(96 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, N_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 96 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#%% training network
if __name__ == '__main__':
    use_gpu = torch.cuda.is_available()
    net = Net()
    if use_gpu:
        net = net.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
#    # save loss curve
#    from tensorboardX import SummaryWriter
#    writer = SummaryWriter('runs')
    
    for epoch in range(ep):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss))
                running_loss = 0.0
                niter = epoch * len(trainloader) + i
#                writer.add_scalar('Train/loss',loss.item(),niter)

    print('Finished Training')
#    writer.export_scalars_to_json('./all_Scalars.json')
#    writer.close()
    
    # save model
    print('Saving Model Parameters...')
    torch.save(net.state_dict(), 'model_weights.pth')
    print('done')
    
