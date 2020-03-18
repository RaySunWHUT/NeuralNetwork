import numpy as np
import h5py

import matplotlib.pyplot as plot


# 加载数据
def load_dataset():

    train_dataset = h5py.File('../../datasets/train_catvnoncat.h5', "r")

    # 可以通过train_dataset.keys()查看键值的集合; [:]: 表示除当前维度以外的所有
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])

    # print(train_set_x_orig[24].shape)

    # your train set labels
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File('../../datasets/test_catvnoncat.h5', "r")

    # your test set features
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])

    # your test set labels
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])

    # the list of classes
    classes = np.array(test_dataset["list_classes"][:])

    # 完善数据维度：由(209,) ---> (1, 209)
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


# 数据预处理
def pre_process_data():

    # 加载数据
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

    # 数据扁平化
    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T

    # shapes = test_set_x_orig.shape
    # test_set_x_flatten = test_set_x_orig.reshape(shapes[0], shapes[1] * shapes[2] * shapes[3]).T
    # 与下面的形式等价, 但更贴近于理解, 即：examples_num * Vector; 转置后: 易于输入神经网络中
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

    # 标准化
    train_set_x = train_set_x_flatten / 255.0
    test_set_x = test_set_x_flatten / 255.0

    return train_set_x, train_set_y, test_set_x, test_set_y


# sigmoid 函数
def sigmoid(z):

    s = 1 / (1 + np.exp(-z))

    return s


# relu 函数

# def relu(z):
#
#     s = np.max(0, z)
#
#     return s


# def main():
#
#     load_dataset()
#
#
# main()

