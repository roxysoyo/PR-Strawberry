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

# print(torch.__version__)

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
# print(hzq_)
loss = nn.CrossEntropyLoss()
optim = torch.optim.SGD(params=hzq_.parameters(), lr=0.01)

for epoch in range(5):
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        output_ = hzq_(imgs)
        # print(output_)
        # print(targets)
        result_loss = loss(output_, targets)
        optim.zero_grad()  # 梯度每一次又要清零，否则会累加
        result_loss.backward()
        optim.step()
        running_loss += result_loss
        # print(result_loss)
    print(running_loss)

print("ok")

# print(output_.shape)
