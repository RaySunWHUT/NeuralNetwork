import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model


def plot_decision_boundary(model, x, y):

    # Set min and max values and give it some padding
    x_min, x_max = x[0, :].min() - 1, x[0, :].max() + 1
    y_min, y_max = x[1, :].min() - 1, x[1, :].max() + 1
    h = 0.01

    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict the function value for the whole grid
    z = model(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)

    # Plot the contour and training examples
    plt.contourf(xx, yy, z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(x[0, :], x[1, :], c=y, cmap=plt.cm.Spectral)


def sigmoid(x):

    s = 1 / (1 + np.exp(-x))

    return s


def load_planar_dataset():

    np.random.seed(1)

    # number of examples
    m = 400

    # number of points per class
    n = int(m / 2)

    # dimensionality
    d = 2

    # data matrix where each row is a single example
    x = np.zeros((m, d))

    # labels vector (0 for red, 1 for blue)
    y = np.zeros((m, 1), dtype='uint8')

    # maximum ray of the flower
    a = 4

    for j in range(2):
        ix = range(n * j, n * (j + 1))
        t = np.linspace(j * 3.12, (j + 1) * 3.12, n) + np.random.randn(n) * 0.2  # theta
        r = a * np.sin(4 * t) + np.random.randn(n) * 0.2  # radius
        x[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j

    x = x.T
    y = y.T

    return x, y


def load_extra_datasets():

    n = 200
    noisy_circles = sklearn.datasets.make_circles(n_samples=n, factor=.5, noise=.3)
    noisy_moons = sklearn.datasets.make_moons(n_samples=n, noise=.2)
    blobs = sklearn.datasets.make_blobs(n_samples=n, random_state=5, n_features=2, centers=6)

    gaussian_quantiles = sklearn.datasets.make_gaussian_quantiles(mean=None, cov=0.5, n_samples=n, n_features=2,
                                                                  n_classes=2, shuffle=True, random_state=None)
    no_structure = np.random.rand(n, 2), np.random.rand(n, 2)

    return noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure

