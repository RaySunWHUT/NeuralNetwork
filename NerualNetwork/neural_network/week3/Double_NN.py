# Package imports
import numpy as np
import matplotlib.pyplot as plot
from testCases_v2 import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

X, Y = load_planar_dataset()

print(X.shape)

print(Y.shape)

Y = np.squeeze(Y)

# Visualize the data:
plot.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plot.cm.Spectral)

plot.show()

clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X.T, np.ravel(Y.T))

# Plot the decision boundary for logistic regression

plot_decision_boundary(lambda x: clf.predict(x), X, Y)
plot.title("Logistic Regression")

plot.show()

# Print accuracy
LR_predictions = clf.predict(X.T)
print('Accuracy of logistic regression: %d ' % float(
    (np.dot(Y, LR_predictions) + np.dot(1 - Y, 1 - LR_predictions)) / float(Y.size) * 100) +
      '% ' + "(percentage of correctly labelled datapoints)")
