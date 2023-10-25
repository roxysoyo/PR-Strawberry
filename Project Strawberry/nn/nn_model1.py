"""
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv1 = nn.Conv2d(1, 20, 5)
        # self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, input):
        output = input + 1
        return output

    # def forward(self, x):
    #     x = F.relu(self.conv1(x))
    #     return F.relu(self.conv2(x))

model = Model()
x = torch.tensor(1.0)
output = model(x)
print(output)
"""

import torch
import torch.nn.functional as F

input_ = torch.tensor(
    [
        [1, 2, 0, 3, 1],
        [0, 1, 2, 3, 1],
        [1, 2, 1, 0, 0],
        [5, 2, 3, 1, 1],
        [2, 1, 0, 1, 1],
    ]
)

# 输入input和卷积核weight是需要四个参数的，所以用reshape，二参变四参
input_ = torch.reshape(input_, shape=(1, 1, 5, 5))

kernel = torch.tensor([[1, 2, 1], [0, 1, 0], [2, 1, 0]])
kernel = torch.reshape(kernel, shape=(1, 1, 3, 3))
print(input_.shape)
print(kernel.shape)

output = F.conv2d(input=input_, weight=kernel, bias=None, stride=1, padding=1)
# padding 填充层
print("output", output)
