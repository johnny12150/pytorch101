import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import pandas as pd

# https://www.kaggle.com/kaerunantoka/titanic-pytorch-nn-tutorial?fbclid=IwAR0BnzlJIkWXOVGW6DCXyHBPQLv7nXUPBjo0p3o4VkwG5hfERkChLxBO29s
train = pd.read_csv('./data/titanic/train.csv')

x_train_cuda = torch.tensor(train, dtype=torch.float32).cuda()
test = torch.utils.data.TensorDataset(x_train_cuda)
test_loader = torch.utils.data.DataLoader(test, batch_size=256, shuffle=False)
