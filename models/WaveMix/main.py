# install the required libraries like wavemix, torchmetrics, lion-pytorch
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import wavemix
from PIL import Image
from einops.layers.torch import Rearrange
from lion_pytorch import Lion
from torchmetrics.classification import Accuracy
from torchsummary import summary
from torchvision.datasets import VisionDataset
from tqdm import tqdm
import argparse
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# use GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_no = torch.cuda.current_device()
print(torch.cuda.get_device_name(device_no))

parser = argparse.ArgumentParser(description='WaveMix')
parser.add_argument('--train_path', default=None,
                    help="Path to the training dataset")
parser.add_argument('--test_path', default=None,
                    help="Path to the testing dataset")
parser.add_argument('--test', action='store_true', default=False,
                    help="Whether we should test the model without training. Weights must be specified.")
parser.add_argument('--saved_model_path', default='./saved_models',
                    help="Path to directory containing saved model weights. Weights from training will be saved there")
parser.add_argument('--model_filename', default='model.pth',
                    help="Weights filename")
parser.add_argument('--cmsuffix', default='',
                    help="Additional suffix added to image filenames of confusion matrix.")
parser.add_argument('--verbose', action='store_true', default=False,
                    help="Whether additional debug information should be printed.")
parser.add_argument('--do_not_rotate_images', action='store_true', default=False,
                    help="Do not perform automatic rotation of images when no train dataset is specified")
args = parser.parse_args()

num_classes = 27
emnist_train_path = '../../resources/datasets/archives/emnist_download/train'
emnist_test_path = '../../resources/datasets/archives/emnist_download/test'
use_custom_train_loader = True
use_custom_test_loader = True
custom_loader_test_path = args.test_path
custom_loader_train_path = args.train_path
test_only = args.test
cmsuffix = args.cmsuffix
debug_print = args.verbose
do_not_rotate_images = args.do_not_rotate_images

if custom_loader_train_path is None:
    use_custom_train_loader = False
if custom_loader_test_path is None:
    use_custom_test_loader = False

if not os.path.exists(args.saved_model_path):
    os.makedirs(args.saved_model_path)

class WaveMix(nn.Module):
    def __init__(
            self,
            *,
            num_classes,
            depth,
            mult,
            ff_channel,
            final_dim,
            dropout,
    ):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                wavemix.Level1Waveblock(mult=mult, ff_channel=ff_channel, final_dim=final_dim, dropout=dropout))

        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Rearrange('... () () -> ...'),
            nn.Linear(final_dim, num_classes)
        )

        self.conv = nn.Sequential(
            nn.Conv2d(1, int(final_dim / 2), 3, 1, 1),
            nn.Conv2d(int(final_dim / 2), final_dim, 3, 1, 1)
        )

    def forward(self, img):
        x = self.conv(img)

        for attn in self.layers:
            x = attn(x) + x

        out = self.pool(x)

        return out


class CustomDataset(VisionDataset):
    """
    This class makes it possible to load a custom dataset.
    """
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
                        [(os.path.join(class_path, f), self.classes.index(class_name) + 1) for f in files])
        else:
            for address, dirs, files in os.walk(root):
                for file in files:
                    dirname = address.split(os.path.sep)[-1]
                    self.file_list.append((os.path.normpath(os.path.join(address, file)), ord(dirname) - 96))

        self.transform = transform

    def __getitem__(self, index):
        file_path, label = self.file_list[index]
        image = Image.open(file_path)

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.file_list)


model = WaveMix(
    num_classes=num_classes,
    depth=16,
    mult=2,
    ff_channel=112,
    final_dim=112,
    dropout=0.5
)


def testOnly(model):
    """
    Test the model without training.
    """
    correct_1 = 0
    correct_5 = 0
    c = 0
    all_preds = []
    all_labels = []

    print("Entering testing mode")
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)

            # Collect all predictions and labels
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            correct_1 += top1_acc(outputs, labels)
            correct_5 += top5_acc(outputs, labels)
            c += 1

    print(f"Top 1: {correct_1 * 100 / c:.2f} - Top 5: {correct_5 * 100 / c:.2f}\n")

    # Generate and display confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=test_dataset.classes,
                yticklabels=test_dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - WaveMix')
    plt.savefig(f'Confusion Matrix - WaveMix - {cmsuffix}.png')
    # plt.show()


model.to(device)
# summary
if debug_print:
    print(summary(model, (1, 28, 28)))
    print(torch.cuda.get_device_properties(device))

# set batch size according to GPU
batch_size = 256

if use_custom_train_loader:

    train_dataset = CustomDataset(custom_loader_train_path,
                                  transform=torchvision.transforms.Compose([
                                      torchvision.transforms.RandomPerspective(),
                                      torchvision.transforms.RandomRotation(10, fill=(0,)),
                                      torchvision.transforms.ToTensor(),
                                      torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                  ]), train=True)
    if debug_print:
        print(f"Number of train classes: {len(train_dataset.classes)}")
        print(train_dataset.classes)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

else:
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.EMNIST(emnist_train_path, split='letters',
                                    train=True, download=True,
                                    transform=torchvision.transforms.Compose([
                                        torchvision.transforms.RandomPerspective(),
                                        torchvision.transforms.RandomRotation(10, fill=(0,)),
                                        torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                    ])),
        batch_size=batch_size, shuffle=True)
    train_dataset = train_loader.dataset
    if debug_print:
        print(f"Number of train classes: {len(train_dataset.classes)}")
        print(train_dataset.classes)

if use_custom_test_loader:

    if use_custom_train_loader or do_not_rotate_images:

        test_dataset = CustomDataset(custom_loader_test_path,
                                     transform=torchvision.transforms.Compose([
                                         torchvision.transforms.ToTensor(),
                                         torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                     ]), train=False)
    else:
        test_dataset = CustomDataset(custom_loader_test_path,
                                     transform=torchvision.transforms.Compose([
                                         torchvision.transforms.RandomRotation([90, 90]),
                                         torchvision.transforms.RandomVerticalFlip(1.0),
                                         torchvision.transforms.ToTensor(),
                                         torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                     ]), train=False)
    if debug_print:
        print(f"Number of test classes: {len(test_dataset.classes)}")
        print(test_dataset.classes)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

else:
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.EMNIST(emnist_test_path, split='letters',
                                    train=False, download=True,
                                    transform=torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                    ])),
        batch_size=batch_size, shuffle=True)
    test_dataset = test_loader.dataset
    if debug_print:
        print(f"Number of test classes: {len(test_dataset.classes)}")
        print(test_dataset.classes)

# Loading the dataset from our own files

# metrics
top1_acc = Accuracy(task="multiclass", num_classes=num_classes).to(device)
top5_acc = Accuracy(task="multiclass", num_classes=num_classes, top_k=5).to(device)

# loss
criterion = nn.CrossEntropyLoss()

# Mixed Precision training
scaler = torch.cuda.amp.GradScaler()

top1 = []
top5 = []
traintime = []
testtime = []
counter = 0

if test_only:
    # load saved model
    model.load_state_dict(torch.load(str(os.path.join(args.saved_model_path, args.model_filename))))
    testOnly(model)
    exit(0)

# Use AdamW or lion as the first optimizer

# optimizer = optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
# print("Training with AdamW")

optimizer = Lion(model.parameters(), lr=1e-4, weight_decay=1e-2)
print("Training with Lion")

# load saved model
# PATH = 'model.pth'
# model.load_state_dict(torch.load(PATH))

epoch = 0
while counter < 20:  # Counter sets the number of epochs of non improvement before stopping

    t0 = time.time()
    epoch_accuracy = 0
    epoch_loss = 0
    running_loss = 0.0
    model.train()

    with tqdm(train_loader, unit="batch") as tepoch:
        tepoch.set_description(f"Epoch {epoch + 1}")

        for data in tepoch:
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            with torch.cuda.amp.autocast():
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            acc = (outputs.argmax(dim=1) == labels).float().mean()
            epoch_accuracy += acc / len(train_loader)
            epoch_loss += loss / len(train_loader)
            tepoch.set_postfix_str(f" loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f}")

    correct_1 = 0
    correct_5 = 0
    c = 0
    model.eval()
    t1 = time.time()
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            correct_1 += top1_acc(outputs, labels)
            correct_5 += top5_acc(outputs, labels)
            c += 1

    print(
        f"Epoch : {epoch + 1} - Top 1: {correct_1 * 100 / c:.2f} - Top 5: {correct_5 * 100 / c:.2f} -  Train Time: {t1 - t0:.2f} - Test Time: {time.time() - t1:.2f}\n")

    top1.append(correct_1 * 100 / c)
    top5.append(correct_5 * 100 / c)
    traintime.append(t1 - t0)
    testtime.append(time.time() - t1)
    counter += 1
    epoch += 1
    if float(correct_1 * 100 / c) >= float(max(top1)):
        torch.save(model.state_dict(), str(os.path.join(args.saved_model_path, args.model_filename)))
        print(1)
        counter = 0

# Second Optimizer
print('Training with SGD')

model.load_state_dict(torch.load(str(os.path.join(args.saved_model_path, args.model_filename))))
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

while counter < 20:  # loop over the dataset multiple times
    t0 = time.time()
    epoch_accuracy = 0
    epoch_loss = 0
    running_loss = 0.0
    model.train()

    with tqdm(train_loader, unit="batch") as tepoch:
        tepoch.set_description(f"Epoch {epoch + 1}")

        for data in tepoch:
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            with torch.cuda.amp.autocast():
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            acc = (outputs.argmax(dim=1) == labels).float().mean()
            epoch_accuracy += acc / len(train_loader)
            epoch_loss += loss / len(train_loader)
            tepoch.set_postfix_str(f" loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f}")

    correct_1 = 0
    correct_5 = 0
    c = 0
    model.eval()
    t1 = time.time()
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            correct_1 += top1_acc(outputs, labels)
            correct_5 += top5_acc(outputs, labels)
            c += 1

    print(
        f"Epoch : {epoch + 1} - Top 1: {correct_1 * 100 / c:.2f} - Top 5: {correct_5 * 100 / c:.2f} -  Train Time: {t1 - t0:.2f} - Test Time: {time.time() - t1:.2f}\n")

    top1.append(correct_1 * 100 / c)
    top5.append(correct_5 * 100 / c)
    traintime.append(t1 - t0)
    testtime.append(time.time() - t1)
    counter += 1
    epoch += 1
    if float(correct_1 * 100 / c) >= float(max(top1)):
        torch.save(model.state_dict(), str(os.path.join(args.saved_model_path, args.model_filename)))
        print(1)
        counter = 0

print('Finished Training')
print("Results")
print(
    f"Top 1 Accuracy: {max(top1):.2f} -Top 5 Accuracy : {max(top5):.2f} - Train Time: {min(traintime):.0f} -Test Time: {min(testtime):.0f}\n")
