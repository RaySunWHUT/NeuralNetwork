import numpy as np
import h5py

import matplotlib.pyplot as plot


def sigmoid(z):

    """
    a -- sigmoid(z)
    cache -- Z, useful during back propagation
    """

    a = 1 / (1 + np.exp(-z))
    cache = z

    return a, cache


def sigmoid_backward(dA, cache):
    """
    backward propagation for a single SIGMOID unit.

    dA -- post - activation gradient
    cache -- 'Z', store for computing backward propagation efficiently

    dz -- Gradient of the cost with respect to Z
    """

    z = cache
    s = 1 / (1 + np.exp(-z))

    dz = dA * s * (1 - s)

    return dz


def relu(z):
    """
    Z -- Output of the linear layer
    A -- Post-activation parameter
    cache -- A stored for computing the backward pass efficiently
    """

    a = np.maximum(0, z)

    cache = z

    return a, cache


def relu_backward(dA, cache):
    """
    dA -- post-activation gradient
    cache -- 'Z', store for computing backward propagation efficiently

    Returns:
    dz -- Gradient of the cost with respect to Z
    """

    z = cache

    # 转换dz到对象
    dz = np.array(dA, copy=True)

    dz[z <= 0] = 0

    return dz


# 加载数据集
def load_dataset():

    train_dataset = h5py.File('../../datasets/train_catvnoncat.h5', "r")

    # print(train_dataset.keys())

    # 可以通过train_dataset.keys()查看键值的集合;
    # [:]: 表示除当前维度以外的所有
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])

    # print("train_set元素个数: " + str(len(train_set_x_orig)))
    # print("train_set图片尺寸: " + str(train_set_x_orig[89].shape))

    # 训练集标签
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File('../../datasets/test_catvnoncat.h5', "r")

    # 测试集特征
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])

    # print("test_set元素个数: " + str(len(test_set_x_orig)))
    # print("test_set图片尺寸: " + str(test_set_x_orig[2].shape))

    # 测试集标签
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])

    # 类别
    classes = np.array(test_dataset["list_classes"][:])

    # 完善数据维度：由(209, ) ---> (1, 209)
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


# 预处理数据函数
def pre_process_data():

    # 加载数据
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

    # 数据扁平化
    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T

    # print(train_set_x_flatten.shape)

    # shapes = test_set_x_orig.shape
    # test_set_x_flatten = test_set_x_orig.reshape(shapes[0], shapes[1] * shapes[2] * shapes[3]).T
    # 与下面的形式等价, 但更贴近于理解, 即：examples_num * Vector; 转置后: 易于输入神经网络中
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

    # 标准化
    train_set_x = train_set_x_flatten / 255.0
    test_set_x = test_set_x_flatten / 255.0

    return train_set_x, train_set_y, test_set_x, test_set_y


def main():

    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

    plot.imshow(train_set_x_orig[25])
    plot.show()

    pre_process_data()


main()
