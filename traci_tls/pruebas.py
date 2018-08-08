import time
import numpy as np
from threading import Thread
from multiprocessing.pool import ThreadPool


def F(n):
    observation = np.zeros((3, 64, 64), dtype=float)

    for i in range(0, 50):
        for x in range(0, 64):
            for y in range(0, 64):
                observation[0, x, y] = 1.0
                observation[1, x, y] = 2.0


def F2(n):
    observation = np.zeros((3, 64, 64), dtype=float)

    for i in range(0, 50):
        for x in range(0, 64):
            for y in range(0, 64):
                observation[0, x, y] = 1.0
                observation[1, x, y] = 2.0


start = time.time()
pool = ThreadPool(processes=1)
threads = []

for ii in range(0, 40):
    threads.append(pool.apply_async(F, (ii,)))

pool.join()

end = time.time()
print("Total time: {}".format(end - start))

start = time.time()
for n in range(0, 40):
    F2(n)
end = time.time()
print("Total time: {}".format(end - start))
