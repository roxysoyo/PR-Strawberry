import torch
import torchvision

vgg16 = torchvision.models.vgg16(pretrained=False)
# 保存方式1：

# torch.save(vgg16, "vgg16_model_method1.pth")

# 保存方式2：
torch.save(vgg16.state_dict(), "vgg16_model_method2.pth")

# 保存方式3：
