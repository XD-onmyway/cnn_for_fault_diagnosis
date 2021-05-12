# import os
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

random.seed(datetime.now())

# ball_18
ball_18_0 = io.loadmat("./data/ball_18/ball_18_0")["X118_DE_time"].tolist()
ball_18_1 = io.loadmat("./data/ball_18/ball_18_1")["X119_DE_time"].tolist()
ball_18_2 = io.loadmat("./data/ball_18/ball_18_2")["X120_DE_time"].tolist()
ball_18_3 = io.loadmat("./data/ball_18/ball_18_3")["X121_DE_time"].tolist()
ball_18 = [ball_18_0, ball_18_1, ball_18_2, ball_18_3]

# ball_36
ball_36_0 = io.loadmat("./data/ball_36/ball_36_0")["X185_DE_time"].tolist()
ball_36_1 = io.loadmat("./data/ball_36/ball_36_1")["X186_DE_time"].tolist()
ball_36_2 = io.loadmat("./data/ball_36/ball_36_2")["X187_DE_time"].tolist()
ball_36_3 = io.loadmat("./data/ball_36/ball_36_3")["X188_DE_time"].tolist()
ball_36 = [ball_36_0, ball_36_1, ball_36_2, ball_36_3]

# ball_54
ball_54_0 = io.loadmat("./data/ball_54/ball_54_0")["X222_DE_time"].tolist()
ball_54_1 = io.loadmat("./data/ball_54/ball_54_1")["X223_DE_time"].tolist()
ball_54_2 = io.loadmat("./data/ball_54/ball_54_2")["X224_DE_time"].tolist()
ball_54_3 = io.loadmat("./data/ball_54/ball_54_3")["X225_DE_time"].tolist()
ball_54 = [ball_54_0, ball_54_1, ball_54_2, ball_54_3]

# inner_18
inner_18_0 = io.loadmat("./data/inner_18/inner_18_0")["X105_DE_time"].tolist()
inner_18_1 = io.loadmat("./data/inner_18/inner_18_1")["X106_DE_time"].tolist()
inner_18_2 = io.loadmat("./data/inner_18/inner_18_2")["X107_DE_time"].tolist()
inner_18_3 = io.loadmat("./data/inner_18/inner_18_3")["X108_DE_time"].tolist()
inner_18 = [inner_18_0, inner_18_1, inner_18_2, inner_18_3]

# inner_36
inner_36_0 = io.loadmat("./data/inner_36/inner_36_0")["X169_DE_time"].tolist()
inner_36_1 = io.loadmat("./data/inner_36/inner_36_1")["X170_DE_time"].tolist()
inner_36_2 = io.loadmat("./data/inner_36/inner_36_2")["X171_DE_time"].tolist()
inner_36_3 = io.loadmat("./data/inner_36/inner_36_3")["X172_DE_time"].tolist()
inner_36 = [inner_36_0, inner_36_1, inner_36_2, inner_36_3]

# inner_54
inner_54_0 = io.loadmat("./data/inner_54/inner_54_0")["X209_DE_time"].tolist()
inner_54_1 = io.loadmat("./data/inner_54/inner_54_1")["X210_DE_time"].tolist()
inner_54_2 = io.loadmat("./data/inner_54/inner_54_2")["X211_DE_time"].tolist()
inner_54_3 = io.loadmat("./data/inner_54/inner_54_3")["X212_DE_time"].tolist()
inner_54 = [inner_54_0, inner_54_1, inner_54_2, inner_54_3]

# outer_18
outer_18_0 = io.loadmat("./data/outer_18/outer_18_0")["X130_DE_time"].tolist()
outer_18_1 = io.loadmat("./data/outer_18/outer_18_1")["X131_DE_time"].tolist()
outer_18_2 = io.loadmat("./data/outer_18/outer_18_2")["X132_DE_time"].tolist()
outer_18_3 = io.loadmat("./data/outer_18/outer_18_3")["X133_DE_time"].tolist()
outer_18 = [outer_18_0, outer_18_1, outer_18_2, outer_18_3]

# outer_36
outer_36_0 = io.loadmat("./data/outer_36/outer_36_0")["X197_DE_time"].tolist()
outer_36_1 = io.loadmat("./data/outer_36/outer_36_1")["X198_DE_time"].tolist()
outer_36_2 = io.loadmat("./data/outer_36/outer_36_2")["X199_DE_time"].tolist()
outer_36_3 = io.loadmat("./data/outer_36/outer_36_3")["X200_DE_time"].tolist()
outer_36 = [outer_36_0, outer_36_1, outer_36_2, outer_36_3]

# outer_54
outer_54_0 = io.loadmat("./data/outer_54/outer_54_0")["X234_DE_time"].tolist()
outer_54_1 = io.loadmat("./data/outer_54/outer_54_1")["X235_DE_time"].tolist()
outer_54_2 = io.loadmat("./data/outer_54/outer_54_2")["X236_DE_time"].tolist()
outer_54_3 = io.loadmat("./data/outer_54/outer_54_3")["X237_DE_time"].tolist()
outer_54 = [outer_54_0, outer_54_1, outer_54_2, outer_54_3]

# normal
normal_0 = io.loadmat("./data/normal/normal_0")["X097_DE_time"].tolist()
normal_1 = io.loadmat("./data/normal/normal_1")["X098_DE_time"].tolist()
normal_2 = io.loadmat("./data/normal/normal_2")["X099_DE_time"].tolist()
normal_3 = io.loadmat("./data/normal/normal_3")["X100_DE_time"].tolist()
normal = [normal_0, normal_1, normal_2, normal_3]

# all_data
all_data = [
    normal,
    ball_18,
    ball_36,
    ball_54,
    inner_18,
    inner_36,
    inner_54,
    outer_18,
    outer_36,
    outer_54,
]
print(len(all_data))


def main(argv=None):
    classes = (
        "normal",
        "ball_18",
        "ball_36",
        "ball_54",
        "inner_18",
        "inner_36",
        "inner_54",
        "outer_18",
        "outer_36",
        "outer_54",
    )

    # classes = ("normal", "error")

    train_pics = []
    train_labels = []
    test_pics = []
    test_labels = []

    for data_type in range(10):
        # 二类
        if data_type == 0:
            the_type = 0
        else:
            the_type = 1
        # 四类
        # the_type = (data_type + 2) // 3
        # 十类
        the_type = data_type
        data = all_data[data_type]
        for load_type in range(4):
            load_data = data[load_type]
            max_start = len(load_data) - 4096
            starts = []
            for i in range(500):
                # 随机一个start，不在starts里，就加入
                while True:
                    start = random.randint(0, max_start)
                    if start not in starts:
                        starts.append(start)
                        break
                # 将4096个数据点转化成64×64的二维图
                temp = load_data[start : start + 4096]
                temp = np.array(temp)
                train_pics.append(temp.reshape(64, 64))
                train_labels.append(the_type)
            for i in range(100):
                while True:
                    start = random.randint(0, max_start)
                    if start not in starts:
                        starts.append(start)
                        break
                temp = load_data[start : start + 4096]
                temp = np.array(temp)
                test_pics.append(temp.reshape(64, 64))
                test_labels.append(the_type)
        print("train_pics", len(train_pics))
        print("train_labels", len(train_labels))
        print("test_pics", len(test_pics))
        print("test_labels", len(test_labels))

    np.savez("train_pics", *train_pics)
    np.savez("train_labels", *train_labels)
    np.savez("test_pics", *test_pics)
    np.savez("test_labels", *test_labels)

    print("save success")


if __name__ == "__main__":
    sys.exit(main())
