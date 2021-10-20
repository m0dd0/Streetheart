import numpy as np
import time

if __name__ == "__main__":
    a = np.random.rand(10_000_000)
    start = time.perf_counter()
    v = 0.6 > a > 0.5
    print(time.perf_counter() - start)
