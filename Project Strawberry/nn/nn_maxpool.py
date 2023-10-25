import torchvision
from torch import nn
import torch.nn.functional as F
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# input_ = torch.tensor([[1, 2, 0, 3, 1],
#                        [0, 1, 2, 3, 1],
#                        [1, 2, 1, 0, 0],
#                        [5, 2, 3, 1, 1],
#                        [2, 1, 0, 1, 1]], dtype=torch.float32)
# print(input_.shape)
# input_ = torch.reshape(input_, shape=(-1, 1, 5, 5))
# print(input_.shape)
dataset = torchvision.datasets.CIFAR10(
    root="./dataset",
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)
dataloader = DataLoader(dataset=dataset, batch_size=64)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=False)

    def forward(self, input):
        output = self.maxpool1(input)
        return output


tudui = Model()
writer = SummaryWriter("../logs")

step = 0
for data in dataloader:
    imgs, targets = data
    output = tudui(imgs)
    print(output.shape)
    writer.add_images("input", img_tensor=imgs, global_step=step)
    # torch.Size([64, 6, 30, 30])
    # output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images("output", img_tensor=output, global_step=step)
    step = step + 1


writer.close()

# tensorboard --logdir=../logs --port=8006
