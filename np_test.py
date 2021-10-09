import numpy as np
import random
import time

sets = [np.ceil(np.random.rand(1000, 2)) for _ in range(10)]
start = time.perf_counter()

myset = set()
for s in sets:
    myset.update(s.tolist())
print(time.perf_counter() - start)
print(len(set))