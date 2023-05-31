import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import dataset2
from rtlsdr import RtlSdr
import scipy.signal as signal
import os
#needs work
train_path = 'training_data'
classes = [d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]
num_classes = len(classes)
data = dataset2.read_train_sets2(train_path, classes, validation_size=0.3)

Xtrain = torch.tensor(data.train.images, dtype=torch.float32)
Ytrain = torch.tensor(data.train.labels, dtype=torch.float32)
Xtest = torch.tensor(data.valid.images, dtype=torch.float32)
Ytest = torch.tensor(data.valid.labels, dtype=torch.float32)

train_dataset = TensorDataset(Xtrain, Ytrain)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
valid_dataset = TensorDataset(Xtest, Ytest)
valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )
        self.fc1 = nn.Linear(4 * 4 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

model = ConvNet(num_classes).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

def train(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

def validate(model, dataloader, criterion):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss
                        running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            _, actual = torch.max(targets.data, 1)
            total += targets.size(0)
            correct += (predicted == actual).sum().item()
    return running_loss / len(dataloader), correct / total

num_epochs = 50

for epoch in range(num_epochs):
    train_loss = train(model, train_loader, criterion, optimizer)
    val_loss, val_acc = validate(model, valid_loader, criterion)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# ============================================== #

print("")
sdr = RtlSdr()
sdr.sample_rate = sample_rate = 2400000
sdr.err_ppm = 56
sdr.gain = 'auto'

correct_predictions = 0

def read_samples(freq):
    f_offset = 250000  # shifted tune to avoid DC
    sdr.center_freq = freq - f_offset
    time.sleep(0.06)
    iq_samples = sdr.read_samples(1221376)
    iq_samples = iq_samples[0:600000]
    fc1 = np.exp(-1.0j * 2.0 * np.pi * f_offset / sample_rate * np.arange(len(iq_samples)))  # shift down 250kHz
    iq_samples = iq_samples * fc1
    return iq_samples

def check(freq, corr):
    samples = []
    iq_samples = read_samples(freq)
    iq_samples = signal.decimate(iq_samples, 48, zero_phase=True)

    real = np.real(iq_samples)
    imag = np.imag(iq_samples)

    iq_samples = np.ravel(np.column_stack((real, imag)))
    iq_samples = iq_samples[:INPUT_DIM]

    samples.append(iq_samples)

    samples = np.array(samples)

    # reshape for convolutional model
    samples = np.reshape(samples, (len(samples), DIM1, DIM2, 2))
    samples = torch.tensor(samples, dtype=torch.float32).to(device)

    prediction = model(samples)

    # print predicted label
    maxim, maxindex = torch.max(prediction.data, 1)
    maxlabel = classes[maxindex.item()]
    print(freq / 1000000, maxlabel, maxim.item() * 100)

    # calculate validation percent
    if corr == maxlabel:
        global correct_predictions
        correct_predictions += 1

check(92900000, "wfm")
check(49250000, "tv")
check(95000000, "wfm")
check(104000000, "wfm")
check(422600000, "tetra")
check(100500000, "wfm")
check(120000000, "other")
check(106300000, "wfm")
check(942200000, "gsm")
check(107800000, "wfm")

sdr.close()

print("Validation:", correct_predictions / 10 * 100)
