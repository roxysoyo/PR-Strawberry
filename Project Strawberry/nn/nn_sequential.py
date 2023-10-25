import torch
from torch import nn
from torch.nn import Linear, Conv2d, MaxPool2d, Flatten, Sequential
from torch.utils.tensorboard import SummaryWriter


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

input_ = torch.ones((64, 3, 32, 32))
output_ = hzq_(input_)
print(output_.shape)

writer = SummaryWriter("../logs_sequential")

writer.add_graph(model=hzq_, input_to_model=input_)
writer.close()

# tensorboard --logdir=../logs_sequential --port=8005
