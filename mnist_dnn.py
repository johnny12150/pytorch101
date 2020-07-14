import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets
from torch.autograd import Variable
Device = torch.device('cuda: 0')
# https://www.itread01.com/content/1550361632.html

# 避免dataloader平行處理失敗
if __name__ == '__main__':
    class LeNet(nn.Module):
        def __init__(self):
            super().__init__()

            self.fc1 = nn.Linear(28*28, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 16)
            self.output = nn.Linear(16, 10)

        # must have
        def forward(self, x):
            h = F.relu(self.fc1(x))
            h = F.relu(self.fc2(h))
            h = F.relu(self.fc3(h))
            return self.output(h)


    classifier = LeNet().to(Device)
    ce_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=.001)

    Batch_size = 256

    train_loader = torch.utils.data.DataLoader(
      torchvision.datasets.MNIST('./data/', train=True, download=True,
                                 transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                     (0.1307,), (0.3081,))
                                 ])),
      batch_size=Batch_size, shuffle=True, num_workers=2)


    testing_data = torchvision.datasets.MNIST('./data/', train=False, download=True,
                                 transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                     (0.1307,), (0.3081,))
                                 ]))

    print(testing_data.test_data.size())
    print(torch.unsqueeze(testing_data.test_data, dim=1).shape)
    print(testing_data.test_data.view(-1, 28*28).shape)
    test_x = torch.unsqueeze(testing_data.test_data, dim=1).float().view(-1, 28*28).to(Device)
    print(test_x.shape)
    test_y = testing_data.test_labels.to(Device)

    for e in range(20):
        for batch_idx, (data, target) in enumerate(train_loader):
            x = data.view(-1, 28*28).to(Device)
            y = target.to(Device)
            classifier.train()
            optimizer.zero_grad()
            pred_y = classifier(x)
            loss = ce_criterion(pred_y, y)
            acc = torch.sum(torch.argmax(pred_y, dim=1) == y).item() / len(y)
            loss.backward()
            optimizer.step()

        classifier.eval()
        with torch.no_grad():
            eval_y = classifier(test_x)
            test_loss = ce_criterion(eval_y, test_y)
            test_acc = torch.sum(torch.argmax(eval_y, dim=1) == test_y).item() / len(test_y)

        print('[Train loss: %.6f, acc: %.2f%%][Test loss: %.6f, acc: %.2f%%]' % (
        loss.item(), acc * 100, test_loss.item(), test_acc * 100))
