import torch
import torchvision
from torch import nn
from torch.nn import Linear, Conv2d, MaxPool2d, Flatten, Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 准备数据集
train_data = torchvision.datasets.CIFAR10(
    root="./dataset",
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)

test_data = torchvision.datasets.CIFAR10(
    root="./dataset",
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度:{}".format(train_data_size))
print("测试数据集的长度:{}".format(test_data_size))

# 利用Dataloader加载数据集
train_dataloader = DataLoader(dataset=train_data, batch_size=64)
test_dataloader = DataLoader(dataset=test_data, batch_size=64)


# 搭建神经网络
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


# 创建网络模型
model_ = Model()
if torch.cuda.is_available():
    model = model_.cuda()

# 损失函数
loss_ = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_ = loss_.cuda()
# 优化器
learning_rate = 1e-2
optimizer_ = torch.optim.SGD(params=model_.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0

# 添加tensorboard
writer = SummaryWriter("../logs_train2")

# 训练的轮数
epoch = 8
for i in range(epoch):
    print("--------第{}轮训练开始--------".format(i + 1))
    # 训练步骤开始
    for data in train_dataloader:
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        output_ = model_(imgs)
        result_loss = loss_(output_, targets)
        # 优化器模型
        optimizer_.zero_grad()
        result_loss.backward()
        optimizer_.step()
        total_train_step += 1
        if total_train_step % 100 == 0:
            print(
                "训练次数：{}，result_loss ={}".format(total_train_step, result_loss.item())
            )
            writer.add_scalar(
                tag="train_loss",
                scalar_value=result_loss.item(),
                global_step=total_train_step,
            )

    # 测试步骤开始
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs = model_(imgs)
            loss = loss_(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    print("整体测试集上的Loss: {}".format(total_test_loss))
    print("整体测试集上的正确率{}".format(total_accuracy / test_data_size))

    writer.add_scalar(
        tag="test_loss", scalar_value=total_test_loss, global_step=total_test_step
    )
    writer.add_scalar(
        tag="test_accuray",
        scalar_value=total_accuracy / test_data_size,
        global_step=total_test_step,
    )
    total_test_step += 1

    torch.save(model_, "model_{}.pth".format(i))
    print("模型已保存")

writer.close()
"""tensorboard --logdir=../logs_train2 --port=8005"""

print("ok")
# print(output_.shape)
