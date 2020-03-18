import numpy as np
import matplotlib.pyplot as plot
from week2.lr_utils import pre_process_data
from week2.lr_utils import sigmoid


# 初始化权重 w 和偏置单元 b 为一定维度的0向量
def initialize_with_zeros(dim):

    # dim, 1 外必须加(), np.zeros(dim, 1): 报错!
    w = np.zeros((dim, 1))

    # 对应偏置单元bias的标量
    b = 0

    return w, b


# 梯度下降辅助函数, 用于激活函数
def propagate(w, b, x, y):
    """
    w -- weights: (num_px * num_px * 3, 1)
    b -- bias, 标量
    x -- 数据规模: (num_px * num_px * 3, number of examples)
    y -- 训练集X对应的label
    """

    # 输入的实例个数
    m = x.shape[1]

    # 计算激活函数
    a = sigmoid(np.dot(w.T, x) + b)

    # 计算代价函数
    cost = -1 / m * np.sum(y * np.log(a) + (1 - y) * np.log(1 - a))

    # 反向传播
    dw = 1 / m * np.dot(x, (a - y).T)
    db = 1 / m * np.sum(a - y)

    # 代价值, 用于后续输出调试
    # np.squeeze(x): 去除x中shape为1的维度
    cost = np.squeeze(cost)

    grads = {"dw": dw,
             "db": db}

    return grads, cost


# 梯度下降: gradient descent algorithm
# 超参数: num_iterations, learning_rate
def optimize(w, b, x, y, num_iterations, learning_rate, print_cost=False):
    """
    w -- weights: (num_px * num_px * 3, 1)
    b -- bias, 标量
    x -- 数据规模: (num_px * num_px * 3, number of examples)
    y -- 训练集X对应的结果
    """

    # 用于绘制学习曲线
    costs = []

    # w, b的梯度值(导数值)
    dw = db = 0

    for i in range(num_iterations):

        # 获取梯度和代价
        grads, cost = propagate(w, b, x, y)

        # 从grads中获取梯度
        dw = grads["dw"]
        db = grads["db"]

        # 同时更新
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # 记录cost代价值
        if i % 100 == 0:
            costs.append(cost)

        # print输出
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}

    # 此处返回的梯度, 不用于后续计算, 仅仅是为了以后展示模型信息
    grads = {"dw": dw,
             "db": db}

    return params, grads, costs


# 利用学习到的参数(w, b) 预测函数
def predict(w, b, x):
    """
    w -- weights: (num_px * num_px * 3, 1)
    b -- bias, 标量
    x -- 数据规模: (num_px * num_px * 3, number of examples)

    y_prediction -- 模型预测的结果向量
    """
    # 输入实例个数
    m = x.shape[1]

    y_prediction = np.zeros((1, m))
    w = w.reshape(x.shape[0], 1)

    # a: 预测概率
    a = sigmoid(np.dot(w.T, x) + b)

    for i in range(a.shape[1]):

        # 根据阈值, 将0 ~ 1之间的值, 转化成0, 1
        if a[0, i] > 0.5:
            y_prediction[0, i] = 1
        else:
            y_prediction[0, i] = 0

    return y_prediction


# 集成所有函数, 构成模型
# 超参数: num_iterations, learning_rate
def model(x_train, y_train, x_test, y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):

    """
    x_train -- (num_px * num_px * 3, m_train)
    y_train -- 训练标签 (vector) of shape (1, m_train)
    x_test --  (num_px * num_px * 3, m_test)
    y_test --  测试标签 (vector) of shape (1, m_test)

    info -- 模型的详细信息
    """

    # 初始化参数
    w, b = initialize_with_zeros(x_train.shape[0])

    # 梯度下降
    parameters, grads, costs = optimize(w, b, x_train, y_train, num_iterations, learning_rate, print_cost)

    # 获得W, b
    w = parameters["w"]
    b = parameters["b"]

    # 预测的训练集 / 测试集的结果
    y_prediction_test = predict(w, b, x_test)
    y_prediction_train = predict(w, b, x_train)

    # print精度
    # np.mean: 求平均值
    print("train accuracy: {}%".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))
    print("test accuracy: {}%".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))

    info = {"costs": costs,
            "Y_prediction_test": y_prediction_test,
            "y_prediction_train": y_prediction_train,
            "w": w,
            "b": b,
            "learning_rate": learning_rate,
            "num_iterations": num_iterations}

    return info


def main():

    train_set_x, train_set_y, test_set_x, test_set_y = pre_process_data()

    info = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005,
                 print_cost=True)

    index = 3

    plot.imshow(test_set_x[:, index].reshape((64, 64, 3)))

    # 不加无法可视化
    plot.show()

    print(info)


main()
