import numpy as np
from classifiers import Classifer


class PA(Classifer):
    def __init__(self):
        Classifer.__init__(self)

    def fit(self, X, Y, epochs=100):
        def update_rule(x, y, y_hat):
            '''
            update rule for passive aggressive algorithm
            :param x: sample
            :param y: true label
            :param y_hat: predicted label
            :return:
            '''
            if y != y_hat:
                tau = self._loss(x, y, y_hat) / x.dot(x.T)
                self.W[y] += (tau * x)
                self.W[y_hat] -= (tau * x)

        return Classifer._fit(self, X, Y, update_rule, epochs)

    def _loss(self, x, y, y_hat):
        return 1 - self.W[y].dot(x.T) + self.W[y_hat].dot(x.T)
