import time
from scipy.spatial import distance
import numpy as np
from matplotlib import pyplot as plt

n_a = 10_000
n_b = 500
metric = "euclidean"

XA = np.random.rand(n_a, 2) * 1000
XB = np.random.rand(n_b, 2) * 1000

plt.scatter(XA[:, 0], XA[:, 1], 2)
plt.show()


start = time.perf_counter()
distance.cdist(XA, XB, metric)
needed_time = time.perf_counter() - start

print(n_a, n_b, metric, needed_time)
