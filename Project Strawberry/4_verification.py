import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import torch
import torchvision
from PIL import Image
from torchvision import transforms
from model_VGG16 import *


def totensor_pic(img_path):
    img = Image.open(img_path)
    # print(img)
    img = img.convert("RGB")
    # 1.Resize

    trans_resize = transforms.Resize((96, 96))

    # 2. convert data types to tensor type
    trans_tensor = transforms.ToTensor()
    transform = torchvision.transforms.Compose([trans_resize, trans_tensor])

    img = transform(img)

    img_ = torch.reshape(img, (1, 3, 96, 96))

    print(img.shape)  # torch.Size([1, 3, 32, 32])
    return img_


def main(img):
    model_classify = torch.load("model_a.pth", map_location="cpu")
    # Transform the model into validation mode
    model_classify.eval()

    # determine the classification
    with torch.no_grad():
        output_classify = model_classify(img)
    # print(output)
    outcome = output_classify.argmax(1)
    i = outcome.item()
    string = ""
    if i == 0:
        string = "Unripe."
        print(string)
    if i == 1:
        string = "Partripe."
        print(string)
    if i == 2:
        string = "Ripe."
        print(string)
    # 2->ripe    1->partripe   0->unripe

    return string


if __name__ == "__main__":
    img_path = "PR1-Strawberry/verification images/v2.PNG"
    img = totensor_pic(img_path)
    main(img)
