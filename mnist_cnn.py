from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# (0.1307, ), (0.3081, ) mean and std deviation of the MNIST dataset
norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307, ), (0.3081, ))])
# load MNIST data
train_loader = DataLoader(datasets.MNIST('./data/', train=True, download=False, transform=norm))
test_loader = DataLoader(datasets.MNIST('./data/', train=False, download=False, transform=norm))


# ref: https://github.com/pytorch/examples/blob/master/mnist/main.py
class Cnn2D(nn.Module):
    def __init__(self):
        super(Cnn2D, self).__init__()  # same as super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train_model(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
          lr +=[ param_group['lr'] ]
    return lr


if torch.cuda.is_available():
    model = Cnn2D().cuda()
else:
    model = Cnn2D()

# https://blog.csdn.net/qyhaill/article/details/103043637
optimizer = optim.Adadelta(model.parameters(), lr=0.001)  # lr decay on iteration
scheduler = StepLR(optimizer, step_size=1, gamma=0.7)  # lr decay on epoch
for epoch in range(1, 30):
    train_model(model, device, train_loader, optimizer, epoch)
    scheduler.step()
