import torch
import torchvision
from torch import nn
from torch.nn import ReLU, Sigmoid, Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(
    root="./dataset",
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)
dataloader = DataLoader(dataset=dataset, batch_size=64, drop_last=True)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = Linear(in_features=196608, out_features=10)
        # self.sigmoid1 =Sigmoid()

    def forward(self, input):
        output = self.linear1(input)
        return output


tudui = Model()

step = 0
for data in dataloader:
    imgs, targets = data
    print(imgs.shape)
    output = torch.flatten(imgs)  # 把张量展成一行
    print(output.shape)
    output = tudui(output)
    print(output.shape)
