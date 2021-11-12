import multiprocessing as mp
import time

import numpy as np


def print_matrix(matrix, rows, cols):
    for i in range(rows):
        for j in range(cols):
            print(round(matrix[i][j], 3), end="    ")
        print()


def fast_expo(a, b):
    if a == 0 and b == 0:
        raise ArithmeticError("0^0 is undefined")
    elif a == 0:
        return 0
    elif b == 1:
        return a
    elif b == 0:
        return 1
    else:
        if b % 2 == 0:
            x = fast_expo(a, b / 2)
            return x * x
        else:
            x = fast_expo(a, b // 2)
            return x * x * a


def unity_roots(n):
    theta = 2 * np.pi * 1j / n

    return [np.exp(theta * i) for i in range(0, n)]


def random_square(seed):
    np.random.seed(seed)
    random_num = np.random.randint(0, 10)
    return random_num**2


def parallel_operation():
    print(f"Number of cpu: {mp.cpu_count()}")

    t0 = time.time()
    n_cpu = mp.cpu_count()

    t0 = time.time()
    results = []
    for i in range(1000000):
        results.append(random_square(i))
    t1 = time.time()
    print(f'Series Execution time {t1 - t0} s')

    t0 = time.time()
    pool = mp.Pool(processes=n_cpu)
    results = [pool.map(random_square, range(1000000))]
    t1 = time.time()
    print(f'Parallel Execution time {t1 - t0} s')


if __name__ == "__main__":
    parallel_operation()
