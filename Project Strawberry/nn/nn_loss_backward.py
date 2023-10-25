import torch
import torchvision
from torch import nn
from torch.nn import (
    Linear,
    Conv2d,
    MaxPool2d,
    Flatten,
    Sequential,
)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(
    root="./dataset",
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)
dataloader = DataLoader(dataset=dataset, batch_size=1)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model1 = Sequential(
            Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            MaxPool2d(kernel_size=2, ceil_mode=False),
            Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            MaxPool2d(kernel_size=2, ceil_mode=False),
            Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            MaxPool2d(kernel_size=2, ceil_mode=False),
            Flatten(),
            Linear(in_features=1024, out_features=64),
            Linear(in_features=64, out_features=10),
        )

    def forward(self, x):
        x = self.model1(x)
        return x


hzq_ = Model()
print(hzq_)

loss = nn.CrossEntropyLoss()

for data in dataloader:
    imgs, targets = data
    output_ = hzq_(imgs)
    print(output_)
    print(targets)
    result_loss = loss(output_, targets)
    result_loss.backward()
    print("ok")
    # print(output_.shape)
