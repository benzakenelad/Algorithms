import numpy as np
from classifiers import Classifer


class Perceptron(Classifer):
    def __init__(self, eta=0.05):
        Classifer.__init__(self, eta)
        self.eta = eta

    def fit(self, X, Y, epochs=100):
        def update_rule(x, y, y_hat):
            '''
            update rule for perceptron algorithm
            :param x: sample
            :param y: true label
            :param y_hat: predicted label
            :return:
            '''
            if y != y_hat:
                self.W[y] = self.W[y] + self.eta * x
                self.W[y_hat] = self.W[y_hat] - self.eta * x

        return Classifer._fit(self, X, Y, update_rule, epochs)
