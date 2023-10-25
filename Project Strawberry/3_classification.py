import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import pandas as pd
import torch
from skimage import io
from torch.utils.data import DataLoader, random_split, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch import nn
from model_VGG16 import *
import math


class StrawDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        # Read the label file
        self.annotations = pd.read_csv(os.path.join(root_dir, csv_file))
        # Define the file directory
        self.root_dir = root_dir
        # Define transform
        self.transform = transform

    # Get the length of the dataset
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # Get the image path
        # 0 representing the first column of the csv file
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        # Read the image
        image = io.imread(img_path)
        # Get the images' labels
        # 1 represents the second column of the csv file
        label = torch.tensor(int(self.annotations.iloc[index, 1]))

        # Apply the transform to the image if in need
        if self.transform:
            image = self.transform(image)

        # Return the image and label
        return image, label


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Give the file name and path
csv_file = "Annotations_copy.csv"
root_dir = "PR1-Strawberry/Images_96"
# Use our own dataset
dataset = StrawDataset(
    csv_file=csv_file, root_dir=root_dir, transform=transforms.ToTensor()
)
len_dataset = len(dataset)
print(len_dataset)
# print(type(dataset))


# Divide the training set and test set
train_ratio, test_ratio = 0.75, 0.25
train_dataset, test_dataset = random_split(
    dataset=dataset,
    lengths=[int(train_ratio * len_dataset), math.ceil(test_ratio * len_dataset)],
    generator=None,
)
train_data_size = len(train_dataset)
test_data_size = len(test_dataset)
print("The length of the training dataset:{}".format(train_data_size))
print("The length of the test dataset:{}".format(test_data_size))

# Load the dataset with Dataloader
batch_size = 128
train_dataloader = DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True
)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


# Build the network
model_ = VGG16()
if torch.cuda.is_available():
    model = model_.cuda()

# loss function
loss_ = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_ = loss_.cuda()

# optimizer
learning_rate = 1e-4
# optimizer_ = torch.optim.SGD(params=model_.parameters(), lr=learning_rate)
optimizer_ = torch.optim.Adam(params=model_.parameters(), lr=learning_rate)

# set some parameters
# record the number of train sessions
total_train_step = 0
# record the number of test sessions
total_test_step = 0
# add tensorboard to visualize the network's learning process
writer = SummaryWriter("../logs")

# training epoch
epoch = 40
for i in range(epoch):
    print("--------NO.{} round of training begins--------".format(i + 1))
    # Start training step
    for data in train_dataloader:
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        output_ = model_(imgs)
        result_loss = loss_(output_, targets)
        # Optimizer model
        optimizer_.zero_grad()
        result_loss.backward()
        optimizer_.step()
        total_train_step += 1
        if total_train_step % 100 == 0:
            print(
                "Training times:{},result_loss ={}".format(
                    total_train_step, result_loss.item()
                )
            )
            writer.add_scalar(
                tag="train_loss",
                scalar_value=result_loss.item(),
                global_step=total_train_step,
            )

    # Start test step
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

    print("Loss on the test dataset: {}".format(total_test_loss))
    print("Accuracy rate on the test dataset{}".format(total_accuracy / test_data_size))

    writer.add_scalar(
        tag="test_loss", scalar_value=total_test_loss, global_step=total_test_step
    )
    writer.add_scalar(
        tag="test_accuracy",
        scalar_value=total_accuracy / test_data_size,
        global_step=total_test_step,
    )
    total_test_step += 1

    torch.save(model_, "model_{}.pth".format(i))
    print("Model saved")

writer.close()
"""
tensorboard usage:
tensorboard --logdir=../logs --port=8006
tensorboard --logdir="logs
"""
print("ok")
