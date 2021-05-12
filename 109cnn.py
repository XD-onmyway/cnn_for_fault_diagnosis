import os
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset
import scipy.io as io
import random
from datetime import datetime


os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(4 * 4 * 256, 2560)
        self.fc2 = nn.Linear(2560, 2)
        # self.fc2 = nn.Linear(2560, 4)

    def forward(self, x):

        # print(x.size())
        x = self.pool(F.relu(self.conv1(x)))
        # print(x.size())
        x = self.pool(F.relu(self.conv2(x)))
        # print(x.size())
        x = self.pool(F.relu(self.conv3(x)))
        # print(x.size())
        x = self.pool(F.relu(self.conv4(x)))
        # print(x.size())
        x = x.view(-1, 4 * 4 * 256)
        # print(x.size())
        x = F.relu(self.fc1(x))
        # print(x.size())
        x = self.fc2(x)
        # print(x.size())
        return x


PATH = "cnn_net.pth"
net = Net()
net.load_state_dict(torch.load(PATH, map_location="cpu"))
# net = Net().to(device)
# net.load_state_dict(torch.load(PATH))
print("load success")


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

train_pics_dict = np.load("train_pics.npz")
train_labels_dict = np.load("train_labels.npz")
test_pics_dict = np.load("test_pics.npz")
test_labels_dict = np.load("test_labels.npz")

# print(test_labels_dict["arr_" + str(3000)])


train_pics = []
train_labels = []
test_pics = []
test_labels = []

# for i in train_pics_dict.files:
#     train_pics.append(train_pics_dict[i])
#     train_labels.append(int(train_labels_dict[i]))

for i in test_pics_dict.files:
    test_pics.append(test_pics_dict[i])
    test_labels.append(int(test_labels_dict[i]))


# print(test_labels)


class MyData(Dataset):
    def __init__(self, pics, labels):
        self.pics = pics
        self.labels = labels

        # print(len(self.pics.files))
        # print(len(self.labels.files))

    def __getitem__(self, index):
        # print(index)
        # print(len(self.pics))
        assert index < len(self.pics)
        return torch.Tensor([self.pics[index]]), self.labels[index]

    def __len__(self):
        return len(self.pics)

    def get_tensors(self):
        return torch.Tensor([self.pics]), torch.Tensor(self.labels)


def main(argv=None):
    # classes = (
    #     "normal",
    #     "ball_18",
    #     "ball_36",
    #     "ball_54",
    #     "inner_18",
    #     "inner_36",
    #     "inner_54",
    #     "outer_18",
    #     "outer_36",
    #     "outer_54",
    # )
    classes = ["normal", "error"]
    # classes = ["normal", "ball", "inner", "outer"]

    # 加载训练数据库

    # trainset = MyData(train_pics, train_labels)
    # trainloader = torch.utils.data.DataLoader(
    #     trainset, batch_size=4, shuffle=True, num_workers=2
    # )

    testset = MyData(test_pics, test_labels)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=True, num_workers=2
    )
    # train
    # for epoch in range(10):
    #     running_loss = 0
    #     for i, data in enumerate(trainloader):
    #         inputs, labels = data
    #         inputs = inputs.cuda()
    #         labels = labels.cuda()
    #         outputs = net(inputs)
    #         loss = criterion(outputs, labels)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #         running_loss += loss
    #         if i % 2000 == 1999:
    #             print(
    #                 "epoch:",
    #                 epoch,
    #                 "[",
    #                 i - 1999,
    #                 ":",
    #                 i,
    #                 "] loss:",
    #                 running_loss.item() / 2000,
    #             )
    #             running_loss = 0
    #     PATH = "cnn_net.pth"
    #     torch.save(net.state_dict(), PATH)
    #     print("save success")

    # test
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            # inputs = inputs.cuda()
            # labels = labels.cuda()
            outputs = net(inputs)
            _, predicts = torch.max(outputs, 1)
            total += 4
            correct += (predicts == labels).sum().item()
    print(correct / total * 100)


if __name__ == "__main__":
    sys.exit(main())
