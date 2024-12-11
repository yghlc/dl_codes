

from __future__ import print_function
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Choose a dataset -- CIFAR10 for example
dataset = datasets.CIFAR10(
    root='data',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.491, 0.482, 0.447],
                             std=[0.247, 0.244, 0.262])
    ])
)

# Create a data loader
train_loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=1)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()  # Call parent class's constructor

        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((8, 8))  # Ensures output size is (8, 8)
        self.fc = nn.Linear(8 * 8 * 8, 10)

    def forward(self, x):
        x = self.conv(x)  # When a nn.Module is called, it will compute the result
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Reshape from (N, C, H, W) to (N, CxHxW)
        x = self.fc(x)
        return x

model = SimpleCNN().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), 0.01,
                            momentum=0.9, weight_decay=5e-4)

def train(epoch):
    model.train()  # Set the model to be in training mode
    for batch_index, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        if batch_index % 10 == 0:
            print('epoch {}  batch {}/{}  loss {:.3f}'.format(
                epoch, batch_index, len(train_loader), loss.item()))

        # Backward
        optimizer.zero_grad()  # Set parameter gradients to zero
        loss.backward()        # Compute (or accumulate, actually) parameter gradients
        optimizer.step()       # Update the parameters

if __name__ == "__main__":
    for epoch in range(1, 6):
        train(epoch)
