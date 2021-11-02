import ssl
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import urllib.request

from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models
from PIL import Image

import pre_processing as pp
import config as cg

ssl._create_default_https_context = ssl._create_unverified_context
r = urllib.request.urlopen('https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth')
resnext50 = models.resnext50_32x4d(pretrained=True)
fc_inputs = resnext50.fc.in_features
resnext50.fc = nn.Linear(fc_inputs, 200)


class LoadDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __getitem__(self, index):
        pair = self.data[index]
        if type(pair) != str:
            img_name, label = pair
            img_path = cg.train_dir + '/' + img_name
        else:
            img_name = pair
            img_path = cg.test_dir + '/' + img_name

        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if type(pair) != str:
            label = int(label) - 1
            label = torch.tensor(label)
            return img, label
        else:
            return img

    def __len__(self):
        return len(self.data)


# load data
all_data = pp.classify_data(txt=cg.data_dir + '/training_labels.txt')
VAL_RATIO = 0.2
percent = int(len(all_data) * (1 - VAL_RATIO))

# separate data
train_data = LoadDataset(all_data[:percent], transform=pp.image_transforms()['train'])
train_data_loader = DataLoader(train_data, batch_size=cg.batch_size, shuffle=True)
valid_data = LoadDataset(all_data[percent:], transform=pp.image_transforms()['valid'])
valid_data_loader = DataLoader(valid_data, batch_size=cg.batch_size, shuffle=False)

# set model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = resnext50.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=cg.lr, momentum=cg.momentum)


def train_and_valid(model, criterion, optimizer, epochs):
    history = []
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        epoch_start = time.time()
        print('Epoch: {}/{}'.format(epoch + 1, epochs))

        model.train()

        train_loss, train_acc, valid_loss, valid_acc = 0.0, 0.0, 0.0, 0.0

        # training
        for i, (inputs, labels) in enumerate(train_data_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

            _, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))

            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            train_acc += acc.item() * inputs.size(0)

        with torch.no_grad():
            model.eval()

        # validating
        for j, (inputs, labels) in enumerate(valid_data_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, labels)

            valid_loss += loss.item() * inputs.size(0)

            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))

            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            valid_acc += acc.item() * inputs.size(0)

        avg_train_loss = train_loss / train_data.__len__()
        avg_train_acc = train_acc / train_data.__len__()

        avg_valid_loss = valid_loss / valid_data.__len__()
        avg_valid_acc = valid_acc / valid_data.__len__()

        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])

        if best_acc < avg_valid_acc:
            best_acc = avg_valid_acc
            best_epoch = epoch + 1

        epoch_end = time.time()

        print(
            "Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, "
            "\n\tValidation: Loss: {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(
                epoch + 1, avg_train_loss, avg_train_acc * 100, avg_valid_loss, avg_valid_acc * 100,
                epoch_end - epoch_start
            ))
        print("Best Accuracy for validation: {:.4f} at epoch {:03d}".format(best_acc, best_epoch))
        print('=======================================================================================')
        torch.save(model, 'models/' + str(epoch + 1) + '.pt')
    return model, history


trained_model, history = train_and_valid(model, criterion, optimizer, cg.epochs)
torch.save(history, 'models/' + 'history.pt')

history = np.array(history)
plt.plot(history[:, 0:2])
plt.legend(['Train Loss', 'Valid Loss'])
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.ylim(0, 1)
plt.savefig('loss_curve.png')

plt.plot(history[:, 2:4])
plt.legend(['Train Loss', 'Valid Loss'])
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.savefig('accuracy_curve.png')

all_data = pp.classify_data(txt=cg.data_dir + '/testing_img_order.txt')
test_data = LoadDataset(all_data, transform=pp.image_transforms['valid'])
test_data_loader = DataLoader(test_data, batch_size=cg.batch_size, shuffle=False)
model = resnext50.to(device)
torch.load('models/{}.pt'.format(cg.epochs))

predict = []
model.eval()  # set the model to evaluation mode
with torch.no_grad():
    for i, img in enumerate(test_data_loader):
        inputs = img
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, test_pred = torch.max(outputs, 1)  # get the index of the class with the highest probability

        for y in test_pred.cpu().numpy():
            predict.append(y + 1)

with open('test_pred.txt', 'w') as f:
    for i in range(len(predict)):
        f.write(str(i) + ' ' + str(predict[i]) + '\n')

with open('./birds/testing_img_order.txt', 'r') as file:
    answer = open('answer.txt', 'w')
    label = []
    for y in predict:
        label.append(pp.find_class_name(y))

    i = 0
    for line in file:
        line = line.strip('\n')
        answer.write(line + ' ' + label[i] + '\n')
        i = i + 1
