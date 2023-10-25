from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, BatchNorm2d, SELU, Dropout, Linear


class CNN_3(nn.Module):
    def __init__(self):
        super(CNN_3, self).__init__()
        self.conv = Sequential(
            Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1),  # 卷积层
            BatchNorm2d(num_features=64),  # 归一化
            SELU(inplace=True),  # 激活函数
            # output(bitch_size, 64, 24, 24)
            MaxPool2d(kernel_size=2, stride=2),  # 最大值池化

            Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1),
            BatchNorm2d(num_features=128),
            SELU(inplace=True),
            # output:(bitch_size, 128, 12 ,12)
            MaxPool2d(kernel_size=2, stride=2),

            Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1),
            BatchNorm2d(num_features=256),
            SELU(inplace=True),
            # output:(bitch_size, 256, 6 ,6)
            MaxPool2d(kernel_size=2, stride=2),
        )
        self.line = Sequential(
            Dropout(p=0.2),
            Linear(in_features=256 * 6 * 6, out_features=4096),
            SELU(inplace=True),
            Dropout(p=0.5),
            Linear(in_features=4096, out_features=1024),
            SELU(inplace=True),
            Linear(in_features=1024, out_features=256),
            SELU(inplace=True),
            Linear(in_features=256, out_features=3),

        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.line(x)
        return x


class avg_CNN_3(nn.Module):
    def __init__(self):
        super(avg_CNN_3, self).__init__()
        self.conv = Sequential(
            Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1),  # 卷积层
            BatchNorm2d(num_features=64),  # 归一化
            SELU(inplace=True),  # 激活函数
            # output(bitch_size, 64, 24, 24)
            MaxPool2d(kernel_size=2, stride=2),  # 最大值池化

            Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1),
            BatchNorm2d(num_features=128),
            SELU(inplace=True),
            # output:(bitch_size, 128, 12 ,12)
            MaxPool2d(kernel_size=2, stride=2),

            Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1),
            BatchNorm2d(num_features=256),
            SELU(inplace=True),
            # output:(bitch_size, 256, 6 ,6)
            MaxPool2d(kernel_size=2, stride=2),
        )
        self.line = Sequential(
            Dropout(p=0.2),
            Linear(in_features=256 * 6 * 6, out_features=4096),
            SELU(inplace=True),
            Dropout(p=0.5),
            Linear(in_features=4096, out_features=1024),
            SELU(inplace=True),
            Linear(in_features=1024, out_features=256),
            SELU(inplace=True),
            Linear(in_features=256, out_features=1),

        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.line(x)
        x = x.squeeze(-1)
        return x
