# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import pytest
from src.det import echlon_form
from typing import Union


def solve(matrix, y):
    size = len(y)

    assert len(y) == size
    assert len(matrix) == size and len(matrix[0]) == size

    sol = [0 for _ in range(size)]

    augmented_matrix = [[0 for _ in range(size + 1)] for _ in range(size)]

    for i in range(size):
        for j in range(size):
            augmented_matrix[i][j] = matrix[i][j]

    for i in range(size):
        augmented_matrix[i][size] = y[i]

    augmented_matrix = echlon_form(augmented_matrix, size, size + 1)

    for i in range(size - 1, -1, -1):
        prod = [augmented_matrix[i][j] * sol[j] for j in range(size)]

        try:
            sol[i] = (augmented_matrix[i][size] - sum(prod)) / augmented_matrix[i][i]
        except ZeroDivisionError:
            raise ArithmeticError("Inconsistent System can not be solved")

    return sol


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pass

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
