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

# ---- Parameters ----
train_path = 'SIGN/sign_mnist_train.csv'  #Path to training csv
test_path = 'SIGN/sign_mnist_test.csv'    #Path to test csv
N_classes = 26                            #Number of classes


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# ---- Subclass for sign language dataset to use with Dataset and Dataloader
# pytorch classes
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

# -- Create Data loaders for training and test data
transform = transforms.Compose([transforms.ToTensor()])
trainset = SIGN(train_path,28,28,transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=1)
testset = SIGN(test_path,28,28,transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=1)
# Define classes
classes = ('A', 'B', 'C', 'D',
           'E', 'F', 'G', 'H', 'I',
            'J','K','L','M','N','O',
            'P','Q','R','S','T','U','V','W','X','Y','Z')

# Define neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 26)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ---------- functions to show an image -------------
#----------------------------------------------------

# get some random training images
if __name__ == '__main__':
    dataiter = iter(trainloader)
#    print("a")
    images, labels = dataiter.next()
#    print("b")

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


    use_gpu = torch.cuda.is_available()


    net = Net()

    if use_gpu:
        net = net.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


    # ------------ Main training loop --------
    # ----------------------------------------

    for epoch in range(1):  # loop over the dataset multiple times
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
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')


    dataiter = iter(testloader)
    images, labels = dataiter.next()

    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    outputs = net(images)

    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(4)))

   # Calculate the percentage of correct predictions
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (
        100 * correct / total))
       

#    class_correct = list(0. for i in range(N_classes))
#    class_total = list(0. for i in range(N_classes))
#    with torch.no_grad():
#        for data in testloader:
#            images, labels = data
#            outputs = net(images)
#            _, predicted = torch.max(outputs, 1)
#            c = (predicted == labels).squeeze()
#            for i in range(4):
#                label = labels[i]
#                class_correct[label] += c[i].item()
#                class_total[label] += 1
#
#
#    for i in range(N_classes):
#        if i == 9 or i == 25:   # skip J and Z since they are not included
#            pass
#        else:
#            print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
    
from convolution_visualization import FilterVisualizer
    
layer = 40
flter = 265
FV = FilterVisualizer(size=56, upscaling_steps=12, upscaling_factor=1.2)
FV.visualize(layer, flter, blur=5)
    
