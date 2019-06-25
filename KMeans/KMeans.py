import numpy as np


def __main__():
    pass


class KMeans:
    def __init__(self, n_clusters, n_init=10):
        self.n_clusters = n_clusters
        self.clusters = []
        for i in range(n_clusters):
            self.clusters.append(list())
        self.loss = []
        self.n_init = n_init
        self.miu = None
        self.dim = 0
        self.history = []

    def fit(self, X):
        if self.miu is not None:
            raise Exception('cannot train the same instance of KMeans twice')
        if len(X.shape) != 2:
            raise Exception('illegal data dimension')
        n = X.shape[0]
        self.dim = X.shape[1]
        distance_matrix = np.zeros(shape=(n, self.n_clusters))
        for k in range(self.n_init):
            # random initialization
            miu = X[np.random.permutation(n)[:self.n_clusters]]
            losses = []
            prev_loss, loss = 0, -1
            while prev_loss != loss:
                prev_loss = loss
                [self.clusters[i].clear() for i in range(self.n_clusters)]
                for i in range(n):
                    for j in range(self.n_clusters):
                        distance_matrix[i, j] = np.linalg.norm(X[i] - miu[j])
                clusters_indices = [np.argmin(d) for d in distance_matrix]
                for i, c_i in enumerate(clusters_indices):
                    self.clusters[int(c_i)].append(X[i])
                miu = np.zeros(shape=(self.n_clusters, self.dim))
                for i, cluster in enumerate(self.clusters):
                    for x in cluster:
                        miu[i] = np.add(miu[i], x)
                    if len(cluster) != 0:
                        miu[i] = miu[i] / len(cluster)
                loss = self.__loss(X, miu)
                losses.append(loss)
            self.history.append((loss, miu, losses))
        self.history = sorted(self.history, key=lambda pair: pair[0])
        self.miu, self.loss = self.history[0][1], self.history[0][2]

    def predict(self, X):
        n = X.shape[0]
        distance_matrix = np.zeros(shape=(n, self.n_clusters))
        for i in range(n):
            for j in range(self.n_clusters):
                distance_matrix[i, j] = np.linalg.norm(X[i] - self.miu[j])
        clusters_indices = [np.argmin(d) for d in distance_matrix]
        return clusters_indices

    def __loss(self, X, miu):
        loss = 0
        n = X.shape[0]
        for i, cluster in enumerate(self.clusters):
            for sample in cluster:
                loss += np.linalg.norm(sample - miu[i]) ** 2
        return loss / n
