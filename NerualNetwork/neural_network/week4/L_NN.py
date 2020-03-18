# import numpy as np
import matplotlib.pyplot as plot

from dnn_utils import *


# 随机初始化
def initialize_parameters_deep(layer_dims):
    """
    layer_dims -- 网络维度
    """
    # seed值固定, 则生成的随机数固定
    np.random.seed(1)

    parameters = {}

    # 网络层数
    layers = len(layer_dims)

    for l in range(1, layers):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) / np.sqrt(layer_dims[l - 1])

        # 初始化为0的原因: 方便算法自我调节截距的移动方向
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters


def linear_forward(a, w, b):
    """
    linear part of a layer's forward propagation.
    A -- activations from previous layer(input data): (size of previous layer, number of examples)
    W -- weights matrix: (size of current layer, size of previous layer)
    b -- bias vector: (size of the current layer, 1)

    Z -- the input of the activation function(pre-activation parameter)
    cache -- "A", "W" and "b" ; stored for computing the backward pass efficiently
    """

    # z = w.dot(a) + b
    z = np.dot(w, a) + b

    cache = (a, w, b)

    return z, cache


# 因activation不同而存在
def linear_activation_forward(a_prev, w, b, activation):
    """
    forward propagation for the LINEAR -> ACTIVATION layer
    A_prev -- activations from previous layer(input data): (size of previous layer, number of examples)
    W -- weights matrix: (size of current layer, size of previous layer)
    b -- bias vector: (size of the current layer, 1)
    activation -- "sigmoid" or "relu"

    A -- the output of the activation function, also called the post-activation value
    cache -- "linear_cache", "activation_cache";
             stored for computing the backward pass efficiently
    """

    linear_cache = 0
    activation_cache = 0
    a = 0

    if activation == "sigmoid":

        z, linear_cache = linear_forward(a_prev, w, b)
        a, activation_cache = sigmoid(z)

    elif activation == "relu":

        z, linear_cache = linear_forward(a_prev, w, b)
        a, activation_cache = relu(z)

    cache = (linear_cache, activation_cache)

    return a, cache


def L_model_forward(x, parameters):
    """
    forward propagation for the [LINEAR -> RELU] * (layers - 1) -> LINEAR -> SIGMOID computation

    X -- data (input size, number of examples)
    parameters -- output of initialize_parameters_deep()

    AL -- last post-activation value
    caches -- : every cache of linear_relu_forward() (there are layers - 1 of them, indexed from 0 to layers - 2)
                the cache of linear_sigmoid_forward() (there is one, indexed layers - 1)
    """

    caches = []
    a = x

    # 网络层数: 由于有b和w, 所以, // 2
    layers = len(parameters) // 2

    # [LINEAR -> RELU] * (layers - 1)
    for l in range(1, layers):
        a_prev = a
        a, cache = linear_activation_forward(a_prev, parameters['W' + str(l)], parameters['b' + str(l)],
                                             activation="relu")
        caches.append(cache)

    # LINEAR -> SIGMOID
    AL, cache = linear_activation_forward(a, parameters['W' + str(layers)], parameters['b' + str(layers)],
                                          activation="sigmoid")
    caches.append(cache)

    return AL, caches


def compute_cost(AL, y):
    """
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    cost -- cross-entropy cost
    """

    # 标签y数量
    m = y.shape[1]

    # 计算AL, y
    cost = (1.0 / m) * (-np.dot(y, np.log(AL).T) - np.dot(1 - y, np.log(1 - AL).T))

    # 将矩阵转化为数字
    # (e.g. turns [[17]] into 17)
    cost = np.squeeze(cost)

    return cost


def linear_backward(dz, cache):
    """
    linear portion of backward propagation for a single layer (layer l)

    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (a_prev, w, b) coming from the forward propagation in the current layer

    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l - 1), same shape as a_prev
    dW -- Gradient of the cost with respect to w (current layer l), same shape as w
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    a_prev, w, b = cache

    # 输入X的数量
    m = a_prev.shape[1]

    dW = 1.0 / m * np.dot(dz, a_prev.T)

    db = 1.0 / m * np.sum(dz, axis=1, keepdims=True)

    dA_prev = np.dot(w.T, dz)

    return dA_prev, dW, db


# 因activation不同而存在
def linear_activation_backward(dA, cache, activation):
    """
    dA -- post-activation gradient for current layer l
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """

    dA_prev = dW = db = 0

    linear_cache, activation_cache = cache

    if activation == "relu":

        dz = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dz, linear_cache)

    elif activation == "sigmoid":

        dz = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dz, linear_cache)

    return dA_prev, dW, db


def L_model_backward(AL, y, caches):
    """
    backward propagation for the [LINEAR -> RELU] * (layers - 1) -> LINEAR -> SIGMOID group

    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (there are (layers - 1) or them,
                indexes from 0 to layers - 2)
                the cache of linear_activation_forward() with "sigmoid" (there is one, index layers - 1)

    grads -- gradients
             grads["dA" + str(l)] = ...
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ...
    """
    grads = {}

    # 网络层数: 由于有b和w, 所以, // 2
    layers = len(caches)

    # 重塑y
    y = y.reshape(AL.shape)

    # 初始化反向传播
    # 对应位置元素相除
    dAL = - (np.divide(y, AL) - np.divide(1 - y, 1 - AL))

    # Lth layer (SIGMOID -> LINEAR) gradients
    # "AL, Y, caches"
    current_cache = caches[layers - 1]

    # "grads["dAL"], grads["dWL"], grads["dbL"]
    grads["dA" + str(layers - 1)], grads["dW" + str(layers)], \
    grads["db" + str(layers)] = linear_activation_backward(dAL, current_cache, activation="sigmoid")

    for l in reversed(range(layers - 1)):
        # lth layer: (RELU -> LINEAR) gradients

        current_cache = caches[l]

        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache,
                                                                    activation="relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters

    grads -- gradients, output of L_model_backward

    Returns:
    parameters -- updated parameters
                  parameters["W" + str(l)] = ...
                  parameters["b" + str(l)] = ...
    """

    # 网络层数: 由于有b和w, 所以, // 2
    layers = len(parameters) // 2

    # 更新参数w, b
    for l in range(layers):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]

        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters


# L_layer_model
def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):  # lr was 0.009
    """
    L-layer neural network: [LINEAR- > RELU] * (L - 1) -> LINEAR -> SIGMOID

    X -- data, numpy array of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps

    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    costs = []  # keep track of cost

    # 随机初始化
    parameters = initialize_parameters_deep(layers_dims)

    # 梯度下降
    for i in range(0, num_iterations):

        # 前向传播: [LINEAR -> RELU] * (L - 1) -> LINEAR -> SIGMOID
        AL, caches = L_model_forward(X, parameters)

        # 计算代价
        cost = compute_cost(AL, Y)

        # 反向传播
        grads = L_model_backward(AL, Y, caches)

        # 更新参数
        parameters = update_parameters(parameters, grads, learning_rate)

        # print
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

        if print_cost and i % 100 == 0:
            costs.append(cost)

    # 绘制学习曲线
    plot.plot(np.squeeze(costs))
    plot.ylabel('cost')
    plot.xlabel('iterations (per hundreds)')
    plot.title("Learning rate =" + str(learning_rate))

    plot.show()

    return parameters


def predict(x, y, parameters):
    """
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model

    p -- predictions for the given dataset X
    """

    # X的数量
    m = x.shape[1]

    # 数据集X的对应的prediction的维度
    p = np.zeros((1, m))

    # 前向传播
    probas, caches = L_model_forward(x, parameters)

    # 将0 ~ 1 -- 映射 --> 0, 1
    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    print("Accuracy: " + str(np.sum((p == y) / m)))


def main():

    # 4 - layer model
    layers_dims = [12288, 20, 7, 5, 1]

    train_x, train_y, test_x, test_y = pre_process_data()

    parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations=2500, print_cost=True)

    predict(train_x, train_y, parameters)

    predict(test_x, test_y, parameters)


main()
