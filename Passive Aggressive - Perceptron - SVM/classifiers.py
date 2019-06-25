import numpy as np


class Classifer():
    def __init__(self, eta=0.05):
        # weights matrix
        self.W = None
        # learning rate
        self.eta = eta
        # number of samples
        self.n = 0
        # sample's dimension
        self.d = 0
        # number of classes
        self.n_classes = 0
        # highest accuracy measured during the train
        self.highest_acc = 0

    def _fit(self, X, Y, update_rule, epochs=100):
        '''
        fitting a model to the data
        :param X: data
        :param Y: labels
        :param update_rule: update rule function
        :param epochs: number of epochs
        :return: highest_acc
        '''
        # initialization
        X = np.array(X)
        self.n, self.d = X.shape[0], X.shape[1]
        self.n_classes = len(set(Y))
        self.W = np.zeros((self.n_classes, self.d))

        # training
        for ep in range(epochs):
            perm = np.random.permutation(self.n)
            X = X[perm]
            Y = Y[perm]
            for x, y in zip(X, Y):
                u = self.W.dot(x.T)
                y_hat = np.argmax(u)
                update_rule(x, y, y_hat)
            current_acc = acc_score(Classifer.predict(self, X), Y)
            if self.highest_acc < current_acc:
                self.highest_acc = current_acc
        self.highest_acc -= 0.01
        for i in range(epochs * 10):
            perm = np.random.permutation(self.n)
            X = X[perm]
            Y = Y[perm]
            for x, y in zip(X, Y):
                u = self.W.dot(x.T)
                y_hat = np.argmax(u)
                update_rule(x, y, y_hat)
            current_acc = acc_score(Classifer.predict(self, X), Y)
            if self.highest_acc < current_acc:
                break
        return self.highest_acc

    def predict(self, X):
        '''
        given data X predicting every sample's class
        :param X: data
        :return: X's prediction labels
        '''
        prediction = X.dot(self.W.T)
        return np.argmax(prediction, axis=1)


def acc_score(y_true, y_predicted):
    '''
    calculate accuracy score
    :param y_true: X's true labels
    :param y_predicted: X's predicted labels
    :return:
    '''
    return sum(y_true == y_predicted) / len(y_true)
