import numpy as np
from classifiers import Classifer


class SVM(Classifer):
    def __init__(self, eta=0.05, lambdaa=0.001):
        Classifer.__init__(self, eta)
        self.lambdaa = lambdaa

    def fit(self, X, Y, epochs=100):
        def update_rule(x, y, y_hat):
            '''
            update rule for svm algorithm
            :param x: sample
            :param y: true label
            :param y_hat: predicted label
            :return:
            '''
            self.W[:, :-1] = (1 - self.eta * self.lambdaa) * self.W[:, :-1]
            if y != y_hat:
                self.W[y] += self.eta * x
                self.W[y_hat] -= self.eta * x

        return Classifer._fit(self, np.concatenate((X, np.ones((X.shape[0], 1))), axis=1), Y, update_rule, epochs)

    def predict(self, X):
        return Classifer.predict(self, np.concatenate((X, np.ones((X.shape[0], 1))), axis=1))
