import numpy as np


class LinearRegression:
    def __init__(self):
        self.theta = None

        pass

    def fit(self, X, y):
        self.theta = np.linalg.pinv(X.T.dot(X)).dot(X.T.dot(y))
        print(self.theta)
        pass

    def evaluate(self, X, y):
        prediction = X.dot(self.theta)
        prediction = (prediction - y) ** 2
        prediction = np.sqrt(prediction)
        sum = np.sum(prediction)
        print(sum / X.shape[0])
        pass
