from typing import Union
import numpy as np


def echlon_form(A: Union[np.array, list[list]], row: int, col: int):
    for i in range(col - 1):
        for j in range(row - 1, i, -1):
            if A[j][i] == 0:
                continue
            else:
                try:
                    req_ratio = A[j][i] / A[j - 1][i]
                    # A[j] = A[j] - req_ratio*A[j-1]
                except ZeroDivisionError:
                    # A[j], A[j-1] = A[j-1], A[j]
                    for x in range(col):
                        temp = A[j][x]
                        A[j][x] = A[j - 1][x]
                        A[j - 1][x] = temp
                    continue
                for k in range(col):
                    A[j][k] = A[j][k] - req_ratio * A[j - 1][k]
    return A


def build_cofactor_matrix(matrix: Union[np.array, list[list]], rejected_row: int, rejected_col: int, size: int):
    a = [[0 for _ in range(size - 1)] for _ in range(size - 1)]
    row_a = 0
    for i in range(size):
        if i == rejected_row:
            pass
        else:
            col_a = 0
            for j in range(size):
                if j == rejected_col:
                    pass
                else:
                    a[row_a][col_a] = matrix[i][j]
                    col_a += 1
            row_a += 1
    return a


def det(matrix: Union[np.array, list[list]], size: int):
    if size == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    else:
        det_sum = 0
        for i in range(size):
            cofactor_matrix = build_cofactor_matrix(matrix, 0, i, size)
            det_sum += det(cofactor_matrix, size - 1) * matrix[0][i] * pow(-1, i)
        return det_sum
