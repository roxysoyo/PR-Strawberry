import torchvision
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# input_ = torch.tensor([[1, -0.5],
#                        [-1, 3]])
#
# output_ = torch.reshape(input_, (-1, 1, 2, 2))
# print(output_)
# print(output_.shape)
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
        self.relu1 = ReLU(inplace=False)
        self.sigmoid1 = Sigmoid()

    # inplace是否进行原地操作
    def forward(self, input):
        output = self.sigmoid1(input)
        return output


# tudui = Model()
# output_ = tudui(input_)
# print(output_)

tudui = Model()
writer = SummaryWriter("../logs_relu_sig")

step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", img_tensor=imgs, global_step=step)
    output = tudui(imgs)
    print(output.shape)
    writer.add_images("output", img_tensor=output, global_step=step)
    step = step + 1

writer.close()

# tensorboard --logdir=../logs_relu_sig --port=8008
