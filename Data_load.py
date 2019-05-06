from __future__ import print_function, division
import torch
import os
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class SIGN(Dataset):
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

trainset = 'SIGN/sign_mnist_train.csv'
transformations = transforms.Compose([transforms.ToTensor()])
SIGN_train = \
        SIGN(trainset, 28, 28, transformations)

x,x_label = SIGN_train.__getitem__(7)
img=x.view(28,28)*255
img_np = img.numpy()
print(img)
plt.imshow(img_np[:,:])
plt.show(block=True)
