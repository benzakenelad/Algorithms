from KMeans import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import numpy as np

X, y = make_blobs(n_samples=1000, n_features=2, centers=3, center_box=(-15, 15))

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
prediction = kmeans.predict(X)
loss = kmeans.loss
plt.figure(1)
plt.plot(range(len(loss)), loss)
plt.figure(2)
plt.scatter(X[:, 0], X[:, 1], c=prediction)
plt.figure(3)
test = np.random.uniform(-15, 15, size=(5000, 2))
test_prediction = kmeans.predict(test)
plt.scatter(test[:, 0], test[:, 1], c=test_prediction)
plt.show()
