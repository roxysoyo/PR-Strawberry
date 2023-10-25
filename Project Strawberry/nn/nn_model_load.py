import torch
import torchvision

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 模型加载1:
# model = torch.load("vgg16_model_method1.pth")


# 模型加载2:
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_model_method2.pth"))

# model = torch.load("vgg16_model_method2.pth")

print(vgg16)

print(torch.cuda.is_available())
