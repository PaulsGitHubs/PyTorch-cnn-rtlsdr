import dataset
import os
import time
from datetime import timedelta
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

batch_size = 16
num_inputs = 2
train_path = 'training_data'

classes = [d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]
num_classes = len(classes)

data = dataset.read_train_sets(train_path, classes, validation_size=0.3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data.images)

    def __getitem__(self, index):
        return self.data.images[index], self.data.labels[index]

train_dataset = CustomDataset(data.train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

valid_dataset = CustomDataset(data.valid)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)


class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_inputs, 64, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=(2, 3), stride=1, padding=(0, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )
        self.fc1 = nn.Linear(24 * 32 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

model = ConvNet(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

def train(epoch, model, dataloader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, torch.max(labels, 1)[1])
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / (i + 1)

def evaluate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, torch.max(labels, 1)[1])
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(torch.max(labels, 1)[1]).sum().item()
            loss = running_loss / (len(dataloader))
            acc = 100.0 * correct / total
            return loss, acc

def show_progress(epoch, train_loss, val_loss, val_acc):
    msg = "Epoch {0}, Train Loss: {1:.3f}, Val Loss: {2:.3f}, Val Acc: {3:>6.1%}"
    print(msg.format(epoch + 1, train_loss, val_loss, val_acc / 100))

    # Load pre-trained model to continue training, if available
    if os.path.exists('checkpoint.pth'):
        model.load_state_dict(torch.load('checkpoint.pth'))

        num_epochs = 25000 // len(train_loader)

    for epoch in range(num_epochs):
        train_loss = train(epoch, model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, valid_loader, criterion)
        show_progress(epoch, train_loss, val_loss, val_acc)
        torch.save(model.state_dict(), 'checkpoint.pth')
