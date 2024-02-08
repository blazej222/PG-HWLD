# -*- coding: utf-8 -*-
"""
This Script contains the default and Spinal VGG code for EMNIST(Letters).
This code trains both NNs as two different models.
This code randomly changes the learning rate to get a good result.
@author: Dipu
"""
from torchvision.datasets import VisionDataset
import os
from PIL import Image
import torch
import torchvision
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import string
import PIL.ImageOps
import argparse

class CustomDataset(VisionDataset):
    def __init__(self, root, transform=None, train=True):
        super(CustomDataset, self).__init__(root, transform=transform)
        self.train = train
        self.classes = sorted(os.listdir(root))
        self.file_list = []

        if self.train:
            for class_name in self.classes:
                class_path = os.path.join(root, class_name)
                if os.path.isdir(class_path):
                    files = os.listdir(class_path)
                    self.file_list.extend(
                        [(os.path.join(class_path, f), self.classes.index(class_name)+1) for f in files])
        else:
            # self.file_list = [(os.path.join(root, f), 0) for f in os.listdir(root) if
            #                   os.path.isfile(os.path.join(root, f))]
            for address, dirs, files in os.walk(root):
                for file in files:
                    dirname = address.split(os.path.sep)[-1]
                    self.file_list.append((os.path.normpath(os.path.join(address,file)),ord(dirname)-96))

        self.transform = transform

    def __getitem__(self, index):
        file_path, label = self.file_list[index]
        image = Image.open(file_path)
        if(reverse_test_images_colors):
            image = PIL.ImageOps.invert(image) # reverse colors
            image = image.rotate(270)
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.file_list)

parser = argparse.ArgumentParser(description='VGG')
parser.add_argument('--train_path', default=None,
                    help="Path to the training dataset")
parser.add_argument('--test_path', default=None,
                    help="Path to the testing dataset")
parser.add_argument('--test', action='store_true', default=False,
                    help="Whether we should test the model without training. Weights must be specified.")
parser.add_argument('--saved_model_path', default='./saved_models',
                    help="Path to directory containing saved model weights. Weights from training will be saved there")
parser.add_argument('--model1_filename', default='model1.pth',
                    help="Weights filename of normal VGG")
parser.add_argument('--model2_filename', default='model2.pth',
                    help="Weights filename of VGG-Spinal")
parser.add_argument('--verbose', action='store_true', default=False,
                    help="Show additional debug information")
parser.add_argument('--rotate_images', action='store_true', default=False,
                    help="Rotate images from test dataset by 90 degrees left, then flip them vertically to match those from default emnist training set.")
args = parser.parse_args()

num_epochs = 200
batch_size_train = 100 # TODO: Try larger batch size
batch_size_test = 1000
learning_rate = 0.005
momentum = 0.5
log_interval = 500
use_custom_train_loader = True
use_custom_test_loader = True
reverse_test_images_colors = 0  # Not used anymore

debug_print = args.verbose
custom_loader_test_path = args.test_path
custom_loader_train_path = args.train_path
emnist_train_path = '../../resources/datasets/archives/emnist_download/train'
emnist_test_path = '../../resources/datasets/archives/emnist_download/test'
saved_model_path = args.saved_model_path
model1_filename = args.model1_filename
model2_filename = args.model2_filename
test_only = args.test
rotate_images = args.rotate_images

if custom_loader_train_path is None:
    use_custom_train_loader = False
if custom_loader_test_path is None:
    use_custom_test_loader = False

test_dataset = []
train_dataset = []



if use_custom_train_loader:

    train_dataset = CustomDataset(custom_loader_train_path,
                                  transform=torchvision.transforms.Compose([
                                      #torchvision.transforms.RandomPerspective(),
                                      #torchvision.transforms.RandomRotation(10, fill=(0,)),
                                      torchvision.transforms.ToTensor(),
                                      torchvision.transforms.Normalize(
                                          (0.1307,), (0.3081,))
                                  ]), train=True)


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)

else:
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.EMNIST(emnist_train_path, split='letters',
                                    train=True, download=True,
                                    transform=torchvision.transforms.Compose([
                                        torchvision.transforms.RandomPerspective(),
                                        torchvision.transforms.RandomRotation(10, fill=(0,)),
                                        torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(
                                            (0.1307,), (0.3081,))
                                    ])),
        batch_size=batch_size_train, shuffle=True)
    train_dataset = train_loader.dataset

if use_custom_test_loader:

    if use_custom_train_loader:

        test_dataset = CustomDataset(custom_loader_test_path,
                                     transform=torchvision.transforms.Compose([
                                         torchvision.transforms.RandomRotation([90,90]), #apply rotation and flip due to default column-row ordering
                                         torchvision.transforms.RandomVerticalFlip(1.0),
                                         torchvision.transforms.ToTensor(),
                                         torchvision.transforms.Normalize(
                                             (0.1307,), (0.3081,))
                                     ]), train=False)
    else:
        test_dataset = CustomDataset(custom_loader_test_path,
                                     transform=torchvision.transforms.Compose([
                                         torchvision.transforms.ToTensor(),
                                         torchvision.transforms.Normalize(
                                             (0.1307,), (0.3081,))
                                     ]), train=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=True)

else:
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.EMNIST(emnist_test_path, split='letters',
                                    train=False, download=True,
                                    transform=torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(
                                            (0.1307,), (0.3081,))
                                    ])),
        batch_size=batch_size_test, shuffle=True)
    test_dataset = test_loader.dataset

if debug_print:
    for i in range(6):
        image, label = train_dataset[i]
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(image[0], cmap='gray', interpolation='none')
        plt.title("Label: {}".format(label))
        plt.xticks([])
        plt.yticks([])
    plt.show()

    fig = plt.figure()
    for i in range(6):
        image, label = test_dataset[i]
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(image[0], cmap='gray', interpolation='none')
        plt.title("Label: {}".format(label))
        plt.xticks([])
        plt.yticks([])
    plt.show()

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)


print(example_data.shape)

fig = plt.figure()
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Ground Truth: {}".format(example_targets[i]))
  plt.xticks([])
  plt.yticks([])
fig



class VGG(nn.Module):  
    """
    Based on - https://github.com/kkweon/mnist-competition
    from: https://github.com/ranihorev/Kuzushiji_MNIST/blob/master/KujuMNIST.ipynb
    """
    def two_conv_pool(self, in_channels, f1, f2):
        s = nn.Sequential(
            nn.Conv2d(in_channels, f1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f1, f2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        for m in s.children():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        return s
    
    def three_conv_pool(self,in_channels, f1, f2, f3):
        s = nn.Sequential(
            nn.Conv2d(in_channels, f1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f1, f2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f2),
            nn.ReLU(inplace=True),
            nn.Conv2d(f2, f3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        for m in s.children():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        return s
        
    
    def __init__(self, num_classes=62):
        super(VGG, self).__init__()
        self.l1 = self.two_conv_pool(1, 64, 64)
        self.l2 = self.two_conv_pool(64, 128, 128)
        self.l3 = self.three_conv_pool(128, 256, 256, 256)
        self.l4 = self.three_conv_pool(256, 256, 256, 256)
        
        self.classifier = nn.Sequential(
            nn.Dropout(p = 0.5),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.5),
            nn.Linear(512, num_classes),
        )
    
    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)
    
    
Half_width =128
layer_width =128
    
class SpinalVGG(nn.Module):  
    """
    Based on - https://github.com/kkweon/mnist-competition
    from: https://github.com/ranihorev/Kuzushiji_MNIST/blob/master/KujuMNIST.ipynb
    """
    def two_conv_pool(self, in_channels, f1, f2):
        s = nn.Sequential(
            nn.Conv2d(in_channels, f1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f1, f2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        for m in s.children():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        return s
    
    def three_conv_pool(self,in_channels, f1, f2, f3):
        s = nn.Sequential(
            nn.Conv2d(in_channels, f1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f1, f2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f2),
            nn.ReLU(inplace=True),
            nn.Conv2d(f2, f3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        for m in s.children():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        return s
        
    
    def __init__(self, num_classes=62):
        super(SpinalVGG, self).__init__()
        self.l1 = self.two_conv_pool(1, 64, 64)
        self.l2 = self.two_conv_pool(64, 128, 128)
        self.l3 = self.three_conv_pool(128, 256, 256, 256)
        self.l4 = self.three_conv_pool(256, 256, 256, 256)
        
        
        self.fc_spinal_layer1 = nn.Sequential(
            nn.Dropout(p = 0.5), nn.Linear(Half_width, layer_width),
            nn.BatchNorm1d(layer_width), nn.ReLU(inplace=True),)
        self.fc_spinal_layer2 = nn.Sequential(
            nn.Dropout(p = 0.5), nn.Linear(Half_width+layer_width, layer_width),
            nn.BatchNorm1d(layer_width), nn.ReLU(inplace=True),)
        self.fc_spinal_layer3 = nn.Sequential(
            nn.Dropout(p = 0.5), nn.Linear(Half_width+layer_width, layer_width),
            nn.BatchNorm1d(layer_width), nn.ReLU(inplace=True),)
        self.fc_spinal_layer4 = nn.Sequential(
            nn.Dropout(p = 0.5), nn.Linear(Half_width+layer_width, layer_width),
            nn.BatchNorm1d(layer_width), nn.ReLU(inplace=True),)
        self.fc_out = nn.Sequential(
            nn.Dropout(p = 0.5), nn.Linear(layer_width*4, num_classes),)
        
    
    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = x.view(x.size(0), -1)
        
        x1 = self.fc_spinal_layer1(x[:, 0:Half_width])
        x2 = self.fc_spinal_layer2(torch.cat([ x[:,Half_width:2*Half_width], x1], dim=1))
        x3 = self.fc_spinal_layer3(torch.cat([ x[:,0:Half_width], x2], dim=1))
        x4 = self.fc_spinal_layer4(torch.cat([ x[:,Half_width:2*Half_width], x3], dim=1))
        
        x = torch.cat([x1, x2], dim=1)
        x = torch.cat([x, x3], dim=1)
        x = torch.cat([x, x4], dim=1)
        
        x = self.fc_out(x)

        return F.log_softmax(x, dim=1)



    
device = 'cuda' 
    


# For updating learning rate
def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def test_model(model1,model2):
    print('Entering testing mode.')
    model1.load_state_dict(torch.load(os.path.join(saved_model_path,model1_filename)))
    model2.load_state_dict(torch.load(os.path.join(saved_model_path,model2_filename)))
    model1.eval()
    model2.eval()
    with torch.no_grad():
        correct1 = 0
        total1 = 0
        correct2 = 0
        total2 = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model1(images)
            _, predicted = torch.max(outputs.data, 1)
            total1 += labels.size(0)
            correct1 += (predicted == labels).sum().item()

            # # Update letter accuracy statistics
            # for i in range(len(labels)):
            #     label = labels[i].item()
            #     letter = chr(label + 96)
            #     letter_accuracy1[letter]['total'] += 1
            #     if predicted[i] == label:
            #         letter_accuracy1[letter]['correct'] += 1
            #
            outputs = model2(images)
            _, predicted = torch.max(outputs.data, 1)
            total2 += labels.size(0)
            correct2 += (predicted == labels).sum().item()
            #
            # for i in range(len(labels)):
            #     label = labels[i].item()
            #     letter = chr(label + 96)
            #     letter_accuracy2[letter]['total'] += 1
            #     if predicted[i] == label:
            #         letter_accuracy2[letter]['correct'] += 1

        print('Test Accuracy of NN: {} %'.format(100 * correct1 / total1))

        print('Test Accuracy of SpinalNet: {} %'.format(100 * correct2 / total2))

# Train the model
total_step = len(train_loader)
curr_lr1 = learning_rate

curr_lr2 = learning_rate



model1 = VGG().to(device)

model2 = SpinalVGG().to(device)



# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer1 = torch.optim.Adam(model1.parameters(), lr=learning_rate)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=learning_rate) 
  
# Train the model
total_step = len(train_loader)

best_accuracy1 = 0
best_accuracy2 = 0

if not os.path.exists(saved_model_path):
    os.mkdir(saved_model_path)

# Testing logic goes here
if test_only:
    test_model(model1,model2)
    exit(0)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model1(images)
        loss1 = criterion(outputs, labels)

        # Backward and optimize
        optimizer1.zero_grad()
        loss1.backward()
        optimizer1.step()
        
        outputs = model2(images)
        loss2 = criterion(outputs, labels)

        # Backward and optimize
        optimizer2.zero_grad()
        loss2.backward()
        optimizer2.step()

        if i == 499:
            print ("Ordinary Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                   .format(epoch+1, num_epochs, i+1, total_step, loss1.item()))
            print ("Spinal Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                   .format(epoch+1, num_epochs, i+1, total_step, loss2.item()))

    letter_accuracy1 = {letter: {'correct': 0, 'total': 0} for letter in string.ascii_lowercase}
    letter_accuracy2 = {letter: {'correct': 0, 'total': 0} for letter in string.ascii_lowercase}
    # Test the model
    model1.eval()
    model2.eval()
    with torch.no_grad():
        correct1 = 0
        total1 = 0
        correct2 = 0
        total2 = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            
            outputs = model1(images)
            _, predicted = torch.max(outputs.data, 1)
            total1 += labels.size(0)
            correct1 += (predicted == labels).sum().item()

            # # Update letter accuracy statistics
            # for i in range(len(labels)):
            #     label = labels[i].item()
            #     letter = chr(label + 96)
            #     letter_accuracy1[letter]['total'] += 1
            #     if predicted[i] == label:
            #         letter_accuracy1[letter]['correct'] += 1
            #
            outputs = model2(images)
            _, predicted = torch.max(outputs.data, 1)
            total2 += labels.size(0)
            correct2 += (predicted == labels).sum().item()
            #
            # for i in range(len(labels)):
            #     label = labels[i].item()
            #     letter = chr(label + 96)
            #     letter_accuracy2[letter]['total'] += 1
            #     if predicted[i] == label:
            #         letter_accuracy2[letter]['correct'] += 1
    
        
        if best_accuracy1>= correct1 / total1:
            # curr_lr1 = learning_rate*np.asscalar(pow(np.random.rand(1),3))
            curr_lr1 = learning_rate * np.ndarray.item(pow(np.random.rand(1), 3))
            update_lr(optimizer1, curr_lr1)
            print('Test Accuracy of NN: {} % Best: {} %'.format(100 * correct1 / total1, 100*best_accuracy1))
        else:
            best_accuracy1 = correct1 / total1
            net_opt1 = model1
            print('Test Accuracy of NN: {} % (improvement)'.format(100 * correct1 / total1))
            torch.save(model1.state_dict(), os.path.join(saved_model_path,'best_' + model1_filename))
            
        if best_accuracy2>= correct2 / total2:
            # curr_lr2 = learning_rate*np.asscalar(pow(np.random.rand(1),3))
            curr_lr2 = learning_rate * np.ndarray.item(pow(np.random.rand(1), 3))
            update_lr(optimizer2, curr_lr2)
            print('Test Accuracy of SpinalNet: {} % Best: {} %'.format(100 * correct2 / total2, 100*best_accuracy2))
        else:
            best_accuracy2 = correct2 / total2
            net_opt2 = model2
            print('Test Accuracy of SpinalNet: {} % (improvement)'.format(100 * correct2 / total2))
            torch.save(model2.state_dict(), os.path.join(saved_model_path, 'best_' + model2_filename))

    # # Print letter accuracy statistics
    # print('Letter Accuracy for NN:')
    # for letter, accuracy in letter_accuracy1.items():
    #     accuracy_percentage = accuracy['correct'] / accuracy['total'] * 100
    #     print('{}: {} %'.format(letter, accuracy_percentage))
    #
    # print('Letter Accuracy for SpinalNet:')
    # for letter, accuracy in letter_accuracy2.items():
    #     accuracy_percentage = accuracy['correct'] / accuracy['total'] * 100
    #     print('{}: {} %'.format(letter, accuracy_percentage))

        model1.train()
        model2.train()
torch.save(model1.state_dict(),os.path.join(saved_model_path,model1_filename))
torch.save(model2.state_dict(),os.path.join(saved_model_path,model2_filename))