import os
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import models, datasets, transforms
import torch.utils.data as tud
import numpy as np
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from PIL import Image
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
n_class = 4
pretrain = False
epoches = 5
traindata = datasets.ImageFolder(root='./dataset/train/',
                                 transform=transforms.Compose([
                                     transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.RandomHorizontalFlip(),  # 以一定的概率对图像进行水平翻转，增加了数据的多样性，有助于模型的泛化能力。
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                 ]))

testdata = datasets.ImageFolder(root='./dataset/test/',
                                transform=transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.RandomHorizontalFlip(),  # 以一定的概率对图像进行水平翻转，增加了数据的多样性，有助于模型的泛化能力。
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ]))

classes = testdata.classes
print(classes)

model = models.resnet18(pretrained=pretrain)
if pretrain == True:
    for para in model.parameters():
        para.requires_grad = False
# 最后一层全连接层（通常是一个线性层）替换为一个新的线性层
model.fc = nn.Linear(in_features=512, out_features=n_class, bias=True)
model = model.to(device)


def train_model(model, train_loader, loss_fn, optimizer, epoch):
    model.train()
    total_loss = 0.
    total_correct = 0.
    total = 0.
    for idx, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        preds = outputs.argmax(dim=1)
        total_correct += torch.sum(preds.eq(labels))
        total_loss += loss.item() * inputs.size(0)
        total += labels.size(0)

    total_loss = total_loss / total
    accuracy = 100 * total_correct / total
    print("轮次:%4d|训练集损失:%.5f|训练集准确率:%6.2f%%" % (epoch + 1, loss, accuracy))
    return total_loss, accuracy


def test_model(model, test_loader, loss_fn, optimizer, epoch):
    model.train()
    total_loss = 0.
    total_corrects = 0.
    total = 0.
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            preds = outputs.argmax(dim=1)
            total += labels.size(0)
            total_loss += loss.item() * inputs.size(0)
            total_corrects += torch.sum(preds.eq(labels))

        loss = total_loss / total
        accuracy = 100 * total_corrects / total
        print("轮次:%4d|训练集损失:%.5f|训练集准确率:%6.2f%%" % (epoch + 1, loss, accuracy))
        return loss, accuracy


loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
train_loader = DataLoader(traindata, batch_size=24, shuffle=True)
test_loader = DataLoader(testdata, batch_size=24, shuffle=True)
for epoch in range(0, epoches):
    loss1, acc1 = train_model(model, train_loader, loss_fn, optimizer, epoch)
    loss2, acc2 = test_model(model, test_loader, loss_fn, optimizer, epoch)

classes = testdata.classes
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

path = "dataset/test/cat/cat_test_9.jpg"
model.eval()
img = Image.open(path)
img_p = transform(img).unsqueeze(0).to(device)
output = model(img_p)
pred = output.argmax(dim=1).item()
plt.imshow(img)
plt.show()
p = 100 * nn.Softmax(dim=1)(output).detach().cpu().numpy()[0]
print('该图像预测类别为:', classes[pred])

print('类别{}的概率为{:.2f}%，类别{}的概率为{:.2f}%，类别{}的概率为{:.2f}%'.format(classes[0], p[0], classes[1], p[1], classes[2], p[2],classes[3], p[3]))
